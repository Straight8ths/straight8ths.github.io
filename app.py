from pathlib import Path
import lark
from tabulate import tabulate
import edinet_tools
import os
import pandas as pd
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials
import json
import feedparser
import time
from datetime import datetime
import requests
import deepl
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
import hashlib
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pickle
from threading import Thread, Lock
import uuid
from werkzeug.utils import secure_filename
import pymupdf

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

### DIRECTORIES SETUP ###
# Define base directory
BASE_DIR = Path(__file__).resolve().parent

BASE_UPLOAD_DIR = BASE_DIR / "uploads"

DOCUMENT_BUCKETS = {
    "whitepapers": BASE_UPLOAD_DIR / "whitepapers",
    "reports": BASE_UPLOAD_DIR / "reports",
    "news": BASE_UPLOAD_DIR / "news",
}

for path in DOCUMENT_BUCKETS.values():
    path.mkdir(parents=True, exist_ok=True)

### DOCUMENT INGESTION FUNCTIONS ###
jobs_lock = Lock()
JOB_STATUS = {}


ALLOWED_EXTENSIONS = {".pdf", ".txt"}

### KEYS AND CLIENTS SETUP ###
# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

deepl_api_key = os.getenv("DEEPL_API_KEY")
edinet_api_key = os.getenv("EDINET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize EDINET client
client = edinet_tools.EdinetClient()
doc_list = client.get_document_types()

# Initialize DeepL client
deepl_client = deepl.DeepLClient(deepl_api_key)

# Google Sheets authentication
service_account_info = json.loads(google_api_key)
credentials = Credentials.from_service_account_info(
    service_account_info,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
gc = gspread.authorize(credentials)

index_name = "example-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(
    name=index_name
)

# Select text splitter, embedding model, and vector store
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

memory = ConversationBufferMemory(
    return_messages=True
)

llm = ChatOpenAI(
model="gpt-5-mini",
temperature=0.4)



@app.route('/reports_handler', methods=['POST'])
def reports_handler():
    mode = request.form.get("mode", "portfolio")
    ticker = request.form.get("ticker", None)
    translate = request.form.get("translate", False)

    job_id = str(uuid.uuid4())

    # Open a thread
    Thread(target=collect_reports, kwargs={"mode": mode, "ticker": ticker, "translate": translate, "job_id": job_id}).start()

    return jsonify({
        "status": "Reports download started.",
        "job_id": job_id
    })

def collect_reports(mode: str, ticker: str = None, translate: bool = False, job_id: str = None) -> None:
    """Extract recent EDINET filings for companies in our Google Sheet list."""
    JOB_STATUS[job_id] = {
        "status": "running",
        "progress": 0
    }

    # Core URLs
    portfolio_url = "https://docs.google.com/spreadsheets/d/1oiqGL-ijryNwwpIFhwkimNM24plQOqgSJC-36q08MP4"
    topix_url = "https://docs.google.com/spreadsheets/d/1gNHw3SUdScw10vHJicuypg6p_-U2qzMvnbqi133wLPI"

    # Pandas settings
    pd.set_option("mode.copy_on_write", True)

    all_filings_df = pd.DataFrame(client.get_recent_filings(days_back=30))
    all_filings_df["secCode"] = all_filings_df["secCode"].astype(str).str[:4]

    # Load portfolio data from Google Sheets into a dataframe
    portfolio_sheet = gc.open_by_url(portfolio_url).sheet1
    portfolio_data = portfolio_sheet.get_all_values()
    portfolio_df = (
    pd.DataFrame(portfolio_data[1:], columns=portfolio_data[0])
    .iloc[:, 1:]
    .reset_index(drop=True))
    
    # Edit the portfolio dataframe
    portfolio_df.drop(["Weight"], axis=1, inplace=True)
    portfolio_df["Ticker"] = portfolio_df["Ticker"].str[:4]
    portfolio_df.insert(2, "EDINET_ID", portfolio_df["Ticker"].apply(client._resolve_company_identifier))
    
    # Load TOPIX data from Google Sheets into a dataframe
    topix_sheet = gc.open_by_url(topix_url).sheet1
    topix_data = topix_sheet.get_all_values()
    topix_df = (
    pd.DataFrame(topix_data[1:], columns=topix_data[0])
    .iloc[:, 1:]
    .reset_index(drop=True))

    def filter_filings(reference_dataframe):
        filtered_filings_df = all_filings_df[all_filings_df["edinetCode"].isin(reference_dataframe["EDINET_ID"])]
        filtered_filings_df["docType"] = filtered_filings_df["docTypeCode"].map(doc_list)
        filtered_filings_df["Name"] = filtered_filings_df["secCode"].map(topix_df.set_index("Code")["Issue"])
        filtered_filings_df.reset_index(drop=True, inplace=True)
        filtered_filings_df = filtered_filings_df[["Name", "secCode", "edinetCode", "submitDateTime", "docTypeCode", "docType", "docID"]]
        filtered_filings_df = filtered_filings_df.sort_values(by=["Name", "submitDateTime"], ascending=[True, False]).reset_index(drop=True)
        return filtered_filings_df
        

    if mode == "portfolio":
        filtered_filings_df = filter_filings(portfolio_df)

    if mode == "name_and_comps" and ticker is not None and ticker in portfolio_df["Ticker"].values:

        # Get competitors for the given ticker
        comps = portfolio_df.loc[portfolio_df["Ticker"] == ticker, "Competitors":].values.flatten().tolist()
        comps = [comp for comp in comps if
                 len(comp) > 0
                 and " JP" in comp
                 and comp[:4].isnumeric()]
        comps = [comp[:4] for comp in comps]
        name_and_comps = [ticker] + comps
        name_and_comps = pd.DataFrame(name_and_comps, columns=["Ticker"])
        name_and_comps["EDINET_ID"] = name_and_comps["Ticker"].apply(client._resolve_company_identifier)

        filtered_filings_df = filter_filings(name_and_comps)

    # Download and save filings as text files
    error_reports = []
    for row in filtered_filings_df.itertuples(index=False):
        docID = row.docID
        tickercode = row.secCode
        Name = str(row.Name).replace(" ", "_")
        docType = row.docType
        submitDateTime = row.submitDateTime[:10]
        try:
            parsed = client.download_filing(docID, extract_data=True, doc_type_code=None)
            # Send text blocks to a text file
            with open(os.path.expanduser(f"{DOCUMENT_BUCKETS['reports']}/{Name}_[{tickercode}.T]_{submitDateTime}_{docType}_{docID}_text_blocks.txt"), "w", encoding="utf-8") as f:
                for block in parsed.get("text_blocks", []):
                    blockID = block["id"]
                    title = block["title"]
                    content = block["content"]
                    if translate == True:
                        title = deepl_client.translate_text(title, target_lang="EN-US")
                        content = deepl_client.translate_text(content, target_lang="EN-US")
                    f.write(f"{str(blockID)}\n")
                    f.write(f"{str(title)}\n")
                    f.write(f"{str(content)}\n")
                    f.write("\n")
            JOB_STATUS[job_id]["progress"] += 1
        except Exception as e:
            error_reports.append(docID)
    JOB_STATUS[job_id]["status"] = "complete"
    return
    
@app.route('/news_handler', methods=['POST'])
def news_handler():
    report_feed = request.form.get("report_feed", "macro_feeds")
    earliest_date = request.form.get("earliest_date", "2023-01-01")
    translate = request.form.get("translate", False)

    job_id = str(uuid.uuid4())

    # Open a thread
    Thread(target=collect_news, kwargs={"report_feed": report_feed, "earliest_date": earliest_date, "translate": translate, "job_id": job_id}).start()

    return jsonify({
        "status": "News download started.",
        "job_id": job_id
    })

def collect_news(report_feed: str = None, earliest_date = None, translate: bool = False, job_id: str = None) -> None:
    """Collect news from RSS feeds and save to a text file."""
    JOB_STATUS[job_id] = {
        "status": "running",
        "progress": 0
    }

    macro_english = {
        "Yomiuri_Business": "https://japannews.yomiuri.co.jp/business/feed/",
        "Yomiuri_Economy": "https://japannews.yomiuri.co.jp/business/economy/feed",
        "Yomiuri_Companies": "https://japannews.yomiuri.co.jp/business/companies/feed",
        "Yomiuri_Markets": "https://japannews.yomiuri.co.jp/business/markets/feed",
        "Yomiuri_Business_Series": "https://japannews.yomiuri.co.jp/business/business-series/feed",
        "Yomiuri_YIES": "https://japannews.yomiuri.co.jp/business/yies/feed",
        "Yomiuri_Asia_Inside_Review": "https://japannews.yomiuri.co.jp/business/asia-inside-review/feed",
        "Tokyo_Stock_Exchange": "https://www.jpx.co.jp/english/rss/markets_news.xml",
        "Financial_Services_Agency": "https://www.fsa.go.jp/fsaEnNewsList_rss2.xml"
    }
    macro_japanese = {
        "NHK": "https://news.web.nhk/n-data/conf/na/rss/cat5.xml",
        "Asahi_Business": "https://www.asahi.com/rss/asahi/business.rdf",
        "METI_Journal": "https://journal.meti.go.jp/feed/",
        "JETRO": "https://www.jetro.go.jp/rss/japan.xml",
        "Bank_of_Japan": "https://www.boj.or.jp/rss/statistics.xml",
        "PR_Times": "https://prtimes.jp/index.rdf",
    }
    macro_feeds = [macro_english, macro_japanese]
    sector_energy_english = {
        "Global_Energy_Infrastructure": "https://globalenergyinfrastructure.com/rss?feed=news",
        "Global_Energy_Infrastructure_Japan": "https://globalenergyinfrastructure.com/rss?topic=japan",
        "J-Power": "https://www.jpower.co.jp/english/erss/enews.xml",
        "Fuji_Oil": "https://www.fujioil.co.jp/en/rss.xml",
        "Asia_Corporate_News_Energy_Alternatives": "https://www.acnnewswire.com/rss/sector/Energy_Alternatives_acn.xml",
        "Asia_Corporate_News_Energy": "https://www.acnnewswire.com/rss/sector/Alternative_Energy_acn.xml",
        "Asia_Corporate_News_Oil_Gas": "https://www.acnnewswire.com/rss/sector/Oil__Gas_acn.xml"
    }
    sector_energy_japanese = {
        "Asahi_Environment_and_Energy": "https://www.asahi.com/rss/asahi/eco.rdf",
    }
    sector_energy_feeds = [sector_energy_english, sector_energy_japanese]
    sector_materials_english = {
        "Japan_Rubber_Weekly": "https://www.japanrubberweekly.com/feed/",
        "Asia_Corporate_News_Metals_Mining": "https://www.acnnewswire.com/rss/sector/Metals__Mining_acn.xml",
        "Asia_Corporate_News_Chemicals": "https://www.acnnewswire.com/rss/sector/Chemicals_Spec.Chem_acn.xml",
        "Asia_Corporate_News_Manufacturing": "https://www.acnnewswire.com/rss/sector/Manufacturing_acn.xml",
        "Asia_Corporate_News_Print_Packaging": "https://www.acnnewswire.com/rss/sector/Print__Package_acn.xml",
        "Asia_Corporate_News_Materials_Nanotech": "https://www.acnnewswire.com/rss/sector/Materials__Nanotech_acn.xml"
    }
    sector_materials_japanese = {
        "NIMS_General": "https://www.nims.go.jp/news/newsall.xml",
        "NIMS_News": "https://www.nims.go.jp/news/news.xml"
    }
    sector_materials_feeds = [sector_materials_english, sector_materials_japanese]
    sector_industrials_english = {
        "Mitsubishi_Heavy_Industries_News": "https://www.mhi.com/rss/mhi_news.xml",
        "Mitsubishi_Group_News": "https://www.mhi.com/rss/group_news.xml",
        "Japan_Industry_News": "https://www.japanindustrynews.com/feed/",
        "Asia_Corporate_News_Industrials": "https://www.acnnewswire.com/rss/sector/Industrial_acn.xml",
        "Asia_Corporate_News_Aerospace_Defense": "https://www.acnnewswire.com/rss/sector/Aerospace__Defence_acn.xml",
        "Asia_Corporate_News_Airlines": "https://www.acnnewswire.com/rss/sector/Airlines_acn.xml",
        "Asia_Corporate_News_Marine_Offshore": "https://www.acnnewswire.com/rss/sector/Marine__Offshore_acn.xml",
        "Asia_Corporate_News_EVs_Transportation": "https://www.acnnewswire.com/rss/sector/EVs_Transportation_acn.xml",
        "Asia_Corporate_News_Logistics": "https://www.acnnewswire.com/rss/sector/Transport__Logistics_acn.xml",
        "Asia_Corporate_News_Sustainability": "https://www.acnnewswire.com/rss/sector/Sustainablity_acn.xml",
        "Asia_Corporate_News_Agritech": "https://www.acnnewswire.com/rss/sector/Agritech_acn.xml",
        "Asia_Corporate_News_Smart_Cities": "https://www.acnnewswire.com/rss/sector/Smart_Cities_acn.xml"
    }
    sector_industrials_japanese = {
        "MLIT_Notices": "https://www.mlit.go.jp/important.rdf",
        "MLIT_News": "https://www.mlit.go.jp/index.rdf",
    }
    sector_industrials_feeds = [sector_industrials_english, sector_industrials_japanese]
    sector_consumer_discretionary_english = {
        "Asia_Corporate_News_Automotive": "https://www.acnnewswire.com/rss/sector/Automotive_acn.xml",
        "Asia_Corporate_News_Beauty_Skincare": "https://www.acnnewswire.com/rss/sector/Beauty__Skin_Care_acn.xml",
        "Asia_Corporate_News_Fashion_Apparel": "https://www.acnnewswire.com/rss/sector/Fashion__Apparel_acn.xml",
        "Asia_Corporate_News_Travel_Tourism": "https://www.acnnewswire.com/rss/sector/Travel__Tourism_acn.xml",
        "Asia_Corporate_News_Watches_Jewelry": "https://www.acnnewswire.com/rss/sector/Watches__Jewelry_acn.xml",
    }
    sector_consumer_discretionary_japanese = {
    }
    sector_consumer_discretionary_feeds = [sector_consumer_discretionary_english, sector_consumer_discretionary_japanese]
    sector_consumer_staples_english = {
        "Asia_Corporate_News_Food_Beverage": "https://www.acnnewswire.com/rss/sector/Food__Beverage_acn.xml"
    }
    sector_consumer_staples_japanese = {
    }
    sector_consumer_staples_feeds = [sector_consumer_staples_english, sector_consumer_staples_japanese]
    sector_healthcare_english = {
        "Asia_Corporate_News_Medicine": "https://www.acnnewswire.com/rss/sector/Medicine_acn.xml",
        "Asia_Corporate_News_Alternative_Medicine": "https://www.acnnewswire.com/rss/sector/Alternative_acn.xml",
        "Asia_Corporate_News_Biotech": "https://www.acnnewswire.com/rss/sector/BioTech_acn.xml",
        "Asia_Corporate_News_Clinical_Trials": "https://www.acnnewswire.com/rss/sector/Clinical_Trials_acn.xml",
        "Asia_Corporate_News_Healthcare_Pharma": "https://www.acnnewswire.com/rss/sector/Healthcare__Pharm_acn.xml",
        "Asia_Corporate_News_Medtech": "https://www.acnnewswire.com/rss/sector/MedTech_acn.xml"
    }
    sector_healthcare_japanese = {
        "MHLW_News": "https://www.mhlw.go.jp/stf/news.rdf",
        "MHLW_Emergency": "https://www.mhlw.go.jp/stf/kinkyu.rdf",
        "MHLW_Influenza": "https://www.mhlw.go.jp/rss/inful_news.rdf"
    }
    sector_healthcare_feeds = [sector_healthcare_english, sector_healthcare_japanese]
    sector_financials_english = {
        "Asian_Development_Bank_News": "https://feeds.feedburner.com/adb_news",
        "Asian_Development_Bank_Whats_New": "https://feeds.feedburner.com/adb_whatsnew",
        "Mizuho_News": "https://rss2.www.mizuho-fg.co.jp/rss?site=AQ83RXG5&item=7",
        "Mizuho_Information": "https://rss2.www.mizuho-fg.co.jp/rss?site=AQ83RXG5&item=8",
        "Mizuho_Investor_Relations": "https://rss2.www.mizuho-fg.co.jp/rss?site=AQ83RXG5&item=9",
        "SoftBank_Press_Releases": "https://group.softbank/en/news/press/index.rdf",
        "SoftBank_Notices": "https://group.softbank/en/news/info/index.rdf",
        "Asia_Corporate_News_Financials": "https://www.acnnewswire.com/rss/sector/Financial_acn.xml",
        "Asia_Corporate_News_Banking_Insurance": "https://www.acnnewswire.com/rss/sector/Banking__Insurance_acn.xml",
        "Asia_Corporate_News_Cards_Payments": "https://www.acnnewswire.com/rss/sector/Cards__Payments_acn.xml",
        "Asia_Corporate_News_Daily_Finance": "https://www.acnnewswire.com/rss/sector/Daily_Finance_acn.xml",
        "Asia_Corporate_News_Exchanges_Software": "https://www.acnnewswire.com/rss/sector/Exchanges__Software_acn.xml",
        "Asia_Corporate_News_FinTech": "https://www.acnnewswire.com/rss/sector/FinTech_acn.xml",
        "Asia_Corporate_News_Funds_Equities": "https://www.acnnewswire.com/rss/sector/Funds__Equities_acn.xml",
        "Asia_Corporate_News_Legal_Compliance": "https://www.acnnewswire.com/rss/sector/Legal__Compliance_acn.xml",
        "Asia_Corporate_News_Private_Equity_Venture_Capital": "https://www.acnnewswire.com/rss/sector/PE_VC__Alternatives_acn.xml",
        "Asia_Corporate_News_Trade_Finance": "https://www.acnnewswire.com/rss/sector/Trade_Finance_acn.xml"
    }
    sector_financials_japanese = {
        "FSA_News": "https://www.fsa.go.jp/sescReportList_rss2.xml",
        "FSA_Other_News": "https://www.fsa.go.jp/sescOtherList_rss2.xml",
    }
    sector_financials_feeds = [sector_financials_english, sector_financials_japanese]
    sector_information_technology_english = {
        "The_Bridge": "https://thebridge.jp/en/feed",
        "Asia_Corporate_News_Technology": "https://www.acnnewswire.com/rss/sector/Technology_acn.xml",
        "Asia_Corporate_News_Artificial_Intelligence": "https://www.acnnewswire.com/rss/sector/Artificial_Intel_[AI]_acn.xml",
        "Asia_Corporate_News_Automation": "https://www.acnnewswire.com/rss/sector/Automation_[IoT]_acn.xml",
        "Asia_Corporate_News_Cybersecurity": "https://www.acnnewswire.com/rss/sector/CyberSecurity_acn.xml",
        "Asia_Corporate_News_Datacenters_Cloud": "https://www.acnnewswire.com/rss/sector/Datacenter__Cloud_acn.xml",
        "Asia_Corporate_News_Digitalization": "https://www.acnnewswire.com/rss/sector/Digitalization_acn.xml",
        "Asia_Corporate_News_Electronics": "https://www.acnnewswire.com/rss/sector/Electronics_acn.xml",
        "Asia_Corporate_News_Engineering": "https://www.acnnewswire.com/rss/sector/Engineering_acn.xml",
        "Asia_Corporate_News_Enterprise_IT": "https://www.acnnewswire.com/rss/sector/Enterprise_IT_acn.xml"
    }
    sector_information_technology_japanese = {
        "Digital_Agency_News": "https://www.digital.go.jp/rss/news.xml",
    }
    sector_information_technology_feeds = [sector_information_technology_english, sector_information_technology_japanese]
    sector_communication_services_english = {
        "RCR_Wireless": "https://www.rcrwireless.com/feed",
        "Asia_Corporate_News_Advertising": "https://www.acnnewswire.com/rss/sector/Advertising_acn.xml",
        "Asia_Corporate_News_Broadcast_Film": "https://www.acnnewswire.com/rss/sector/Broadcast_Film__Sat_acn.xml",
        "Asia_Corporate_News_Media_Marketing": "https://www.acnnewswire.com/rss/sector/Media__Marketing_acn.xml",
        "Asia_Corporate_News_Telecoms": "https://www.acnnewswire.com/rss/sector/Telecoms_5G_acn.xml",
        "Asia_Corporate_News_Wireless": "https://www.acnnewswire.com/rss/sector/Wireless_Apps_acn.xml"
    }
    sector_communication_services_japanese = {
    }
    sector_communication_services_feeds = [sector_communication_services_english, sector_communication_services_japanese]
    sector_utilities_english = {
        "Asia_Corporate_News_Water": "https://www.acnnewswire.com/rss/sector/Water_acn.xml"
    }
    sector_utilities_japanese = {
    }
    sector_utilities_feeds = [sector_utilities_english, sector_utilities_japanese]
    sector_real_estate_english = {
        "Real_Estate_Japan": "https://resources.realestate.co.jp/feed/",
        "Asia_Corporate_News_Real_Estate": "https://www.acnnewswire.com/rss/sector/Real_Estate__REIT_acn.xml"
    }
    sector_real_estate_japanese = {
        "Japan_Real_Estate_Investment_Corporation": "https://www.j-re.co.jp/ja_cms/news.xml",
    }
    sector_real_estate_feeds = [sector_real_estate_english, sector_real_estate_japanese]
    gics_feeds = [
        sector_energy_feeds,
        sector_materials_feeds,
        sector_industrials_feeds,
        sector_consumer_discretionary_feeds,
        sector_consumer_staples_feeds,
        sector_healthcare_feeds,
        sector_financials_feeds,
        sector_information_technology_feeds,
        sector_communication_services_feeds,
        sector_utilities_feeds,
        sector_real_estate_feeds
    ]
    parameter_list = {
        "macro": macro_feeds,
        "energy": sector_energy_feeds,
        "materials": sector_materials_feeds,
        "industrials": sector_industrials_feeds,
        "consumer_discretionary": sector_consumer_discretionary_feeds,
        "consumer_staples": sector_consumer_staples_feeds,
        "healthcare": sector_healthcare_feeds,
        "financials": sector_financials_feeds,
        "information_technology": sector_information_technology_feeds,
        "communication_services": sector_communication_services_feeds,
        "utilities": sector_utilities_feeds,
        "real_estate": sector_real_estate_feeds,
    }
    
    # Make a file if one does not exist
    filepath = BASE_UPLOAD_DIR / "news" / f"rss_feed_output_{datetime.now().strftime('%Y%m%d')}.txt"

    report_feed = parameter_list[report_feed]


    # Download RSS feeds and write to file
    with open(os.path.expanduser(filepath), "w", encoding="utf-8") as file:
        for feed_group in report_feed:
            for name, url in feed_group.items():
                feed = feedparser.parse(url)
                file.write(f"Feed: {name.upper()}\n\n")

                for entry in feed.entries:
                    formatted_time = None

                    if getattr(entry, "published_parsed", None):
                        formatted_time = time.strftime(
                            "%Y-%m-%d", entry.published_parsed
                        )
                    elif getattr(entry, "updated_parsed", None):
                        formatted_time = time.strftime(
                            "%Y-%m-%d", entry.updated_parsed
                        )

                    if not formatted_time or formatted_time < earliest_date:
                        continue

                    file.write(f"Published: {formatted_time}\n")

                    # Title
                    if getattr(entry, "title", None):
                        title = entry.title
                        if feed_group is report_feed[1] and translate == True:
                            title = deepl_client.translate_text(
                                title, target_lang="EN-US"
                            ).text
                        file.write(f"Title: {title}\n")
                        del title

                    # Summary
                    if getattr(entry, "summary", None):
                        summary = entry.summary
                        if feed_group is report_feed[1] and translate == True:
                            summary = deepl_client.translate_text(
                                summary, target_lang="EN-US"
                            ).text
                        file.write(f"Summary: {summary}\n")
                        del summary

                    # Link
                    if getattr(entry, "link", None):
                        file.write(f"Link: {entry.link}\n")

                    file.write("\n")

                    # Explicitly drop entry reference
                    del entry

                file.write("====================================\n\n")

                # Explicit cleanup per feed
                del feed
                file.flush()
            JOB_STATUS[job_id]["progress"] += 1
    JOB_STATUS[job_id]["status"] = "complete"
    return

def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

@app.route("/job/<job_id>")
def job_status(job_id):
    try:
        with jobs_lock:
            job = JOB_STATUS.get(job_id)

        if not job:
            return jsonify({
                "status": "unknown",
                "error": "Job not found"
            }), 404

        return jsonify(job)

    except Exception as e:
        # This is critical — NEVER let this route throw
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

def get_vector_count(index) -> int:
    stats = index.describe_index_stats()
    return stats.get("total_vector_count", 0)

def load_text_file_safe(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        chunks = []

        with pymupdf.open(path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                try:
                    text = page.get_text()
                    time.sleep(0)  # Be gentle on large PDFs
                    if text:
                        chunks.append(text)
                except Exception:
                    print(f"PDF page {page_num} failed")

        return "\n".join(chunks)

    return path.read_text(encoding="utf-8", errors="ignore")

def ingest_single_document(path: Path, job_id: str, bucket: str):
    text = load_text_file_safe(path)
    if not text.strip():
        return

    docs = splitter.create_documents(
        [text],
        metadatas=[{
            "source": str(path),
            "bucket": bucket,
            "job_id": job_id
        }]
    )

    ids = [
        f"{bucket}_{job_id}_{hashlib.sha256(doc.page_content.encode()).hexdigest()}"
        for doc in docs
    ]

    vectorstore.add_documents(
        documents=docs,
        ids=ids
    )

def ingest_many_documents(paths: list[Path], job_id: str, bucket: str):
    try:
        before_count = get_vector_count(index)

        for path in paths:
            try:
                print(f"[{job_id}] START {path.name}")
                ingest_single_document(path, job_id, bucket)
                print(f"[{job_id}] DONE {path.name}")
                with jobs_lock:
                    JOB_STATUS[job_id]["processed"] += 1
            except Exception as e:
                print(f"[{job_id}] failed {path.name}: {e}")

        after_count = get_vector_count(index)

        JOB_STATUS[job_id]["vectors_added"] = after_count - before_count
        JOB_STATUS[job_id]["status"] = "complete"

    except Exception as e:
        JOB_STATUS[job_id]["status"] = "error"
        JOB_STATUS[job_id]["error"] = str(e)

def ingest_many_documents_safe(items, job_id, bucket):
    try:
        with jobs_lock:
            JOB_STATUS[job_id]["status"] = "running"
            JOB_STATUS[job_id]["processed"] = 0
            JOB_STATUS[job_id]["vectors_added"] = 0
            JOB_STATUS[job_id]["error"] = None

        saved_paths = []

        for item in items:
            if isinstance(item, tuple):
                file, path = item
                file.save(path)
                saved_paths.append(path)
            else:
                saved_paths.append(item)

        ingest_many_documents(saved_paths, job_id, bucket)
        JOB_STATUS[job_id]["status"] = "complete"

        for path in saved_paths:
            if path.parent == DOCUMENT_BUCKETS[bucket]:
                try:
                    path.unlink()
                except Exception as e:
                    print(f"Failed to delete {path}: {e}")
        
        # Also empty the downloads folder
        downloads_dir = BASE_DIR / "downloads"
        for download_file in downloads_dir.iterdir():
            try:
                download_file.unlink()
            except Exception as e:
                print(f"Failed to delete {download_file}: {e}")

    except Exception as e:
        JOB_STATUS[job_id]["status"] = "error"
        JOB_STATUS[job_id]["error"] = str(e)

@app.route("/upload", methods=["POST"])
def upload():
    bucket = request.form.get("bucket", "whitepapers")
    mode = request.form.get("mode", "file")

    if bucket not in DOCUMENT_BUCKETS:
        return jsonify({"error": "Invalid bucket"}), 400

    job_id = str(uuid.uuid4())

    JOB_STATUS[job_id] = {
        "status": "queued",
        "error": None,
        "bucket": bucket,
        "started_at": time.time(),
        "processed": 0,
        "vectors_added": 0
}

    if mode == "file":
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        safe_name = secure_filename(file.filename)
        path = DOCUMENT_BUCKETS[bucket] / f"{job_id}_{safe_name}"

        file.save(path)

        paths = [path]

    elif mode == "directory":
        paths = [
            p for p in DOCUMENT_BUCKETS[bucket].iterdir()
            if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}
        ]

    else:
        return jsonify({"error": "Invalid mode"}), 400

    Thread(
        target=ingest_many_documents_safe,
        args=(paths, job_id, bucket),
        daemon=True
    ).start()

    return jsonify({
        "status": "Ingestion started",
        "job_id": job_id,
        "bucket": bucket,
        "documents": len(paths)
    })

@app.route("/vector_db_status", methods=["GET"])
def vector_db_status_route():
    try:
        stats = index.describe_index_stats()
        # Convert to dictionary for Flask
        stats_dict = stats.to_dict()  # <-- converts to JSON-serializable dict
        return jsonify({
            "status": "healthy",
            "stats": stats_dict
        })
    except Exception as e:
        return jsonify({
            "status": "failed",
            "error": str(e)
        })

@app.route("/clear_vector_db", methods=["POST"])
def clear_vector_db_route():
    try:
        pc.delete_index(name="example-index")
        pc.create_index(
            name="example-index",
            dimension=1536,  # OpenAI text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        return jsonify({"status": "Vector DB cleared."})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}    )

    # Retrieve relevant documents
    docs = retriever.invoke(user_message)
    context = "\n\n".join(f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs)

    # Build prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant at a friendly-activist hedge fund, whose philosophical alignment is based on the contents of the PARA corpus document within your context. Use the information in your context to field user questions, and always filter your answers through the PARA corpus as a lens. If a piece of information is NOT in your context, clearly state that you do not have that information."
        ),
        (
            "system",
            "Context:\n{context}"
        ),
        (
            "human",
            "{question}"
        )
    ])

    # Load conversation history
    history = memory.load_memory_variables({}).get("history", [])

    # Build messages
    messages = prompt.format_messages(
        context=context,
        question=user_message
    )

    # Insert history between system context and current question
    messages = messages[:2] + history + messages[2:]

    # Call LLM
    response = llm.invoke(messages)

    # Save to memory
    memory.save_context(
        {"input": user_message},
        {"output": response.content}
    )

    return jsonify({"response": response.content})

if __name__ == '__main__':
    app.run(debug=True)

# === To delete the Pinecone index, uncomment the following lines ===
# pc.delete_index(name=index_name)
# print("Pinecone index deleted.")

# === To view Pinecone index stats, uncomment the following line ===
# print(index.describe_index_stats())