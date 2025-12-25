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
import pdfplumber
import pickle

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

### DIRECTORIES SETUP ###
# Define base directory
BASE_DIR = Path(__file__).resolve().parent

# Create EDINET data directory
EDINET_DATA_ROOT = Path(os.environ.get("EDINET_DATA_ROOT", BASE_DIR / "EDINET_data"))
EDINET_DATA_ROOT.mkdir(exist_ok=True)

# Create cache directory and file path
EDINET_CACHE_DIR = EDINET_DATA_ROOT / "EDINET_cache"
EDINET_CACHE_DIR.mkdir(exist_ok=True)
cache_file = EDINET_CACHE_DIR / "data_cache.pkl"

# Create reports directory
EDINET_REPORTS_PATH = EDINET_DATA_ROOT / "EDINET_reports"
EDINET_REPORTS_PATH.mkdir(exist_ok=True)

# Create RSS feed output directory
RSS_OUTPUT_PATH = BASE_DIR / "RSS_feed_output"
RSS_OUTPUT_PATH.mkdir(exist_ok=True)

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
model="gpt-4o-mini",  # or gpt-4.1 / gpt-4o
temperature=0.2)

### DOCUMENT INGESTION FUNCTIONS ###
def ingest_document(path: Path):
    def sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def load_text_file(path: Path) -> str:
        if Path(path).suffix.lower() == ".pdf":
            with pdfplumber.open(path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        else:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    
    # 1. Load
    whole_text = load_text_file(path)

    # 2. Document-level hash
    doc_hash = sha256(whole_text)

    # 3. Chunk into Document objects
    docObjects = splitter.create_documents(
        [whole_text],
        metadatas=[{
            "source": str(path),
            "doc_hash": doc_hash}]
    )

    # 4. Generate stable IDs (doc + chunk hash)
    ids = []
    for doc in docObjects:
        chunk_hash = sha256(doc.page_content)
        ids.append(f"{doc_hash}_{chunk_hash}")

    # Check if IDs are already in the index
    existing_ids = set()
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        id_batch = ids[i:i + batch_size]
        query_response = index.fetch(ids=id_batch)
        existing_ids.update(query_response.vectors.keys())

    # Filter out existing documents
    docObjects = [doc for doc, id_ in zip(docObjects, ids) if id_ not in existing_ids]
    ids = [id_ for id_ in ids if id_ not in existing_ids]

    # 5. Upsert
    vectorstore.add_documents(
        documents=docObjects,
        ids=ids
    )

def ingest_directory(folder: str):
    folder_path = Path(folder)

    for path in folder_path.iterdir():
        if path.suffix.lower() in {".txt", ".md"}:
            ingest_document(path)

def edinet_report_downloader(mode: str, ticker: str = None, translate: bool = False) -> None:
    """Extract recent EDINET filings for companies in our Google Sheet list."""    

    # Core URLs
    portfolio_url = "https://docs.google.com/spreadsheets/d/1oiqGL-ijryNwwpIFhwkimNM24plQOqgSJC-36q08MP4"
    topix_url = "https://docs.google.com/spreadsheets/d/1gNHw3SUdScw10vHJicuypg6p_-U2qzMvnbqi133wLPI"

    # Pandas settings
    pd.set_option("mode.copy_on_write", True)

    def fetch_data():
        all_filings_df = pd.DataFrame(client.get_recent_filings(days_back=30))
        all_filings_df["secCode"] = all_filings_df["secCode"].astype(str).str[:4]
        return all_filings_df

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            all_filings_df = pickle.load(f)
    else:
        all_filings_df = fetch_data()
        with open(cache_file, "wb") as f:
            pickle.dump(all_filings_df, f)

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
        comps = [comp for comp in comps if len(comp) > 0 and " JP" in comp]
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
        Name = row.Name.replace(" ", "_")
        docType = row.docType
        submitDateTime = row.submitDateTime[:10]
        try:
            parsed = client.download_filing(docID, extract_data=True, doc_type_code=None)
            # Send text blocks to a text file
            with open(os.path.expanduser(f"{EDINET_REPORTS_PATH}/{Name}_[{tickercode}.T]_{submitDateTime}_{docType}_{docID}_text_blocks.txt"), "w", encoding="utf-8") as f:
                for block in parsed.get("text_blocks", []):
                    blockID = block["id"]
                    title = block["title"]
                    content = block["content"]
                    if translate:
                        title = deepl_client.translate_text(title, target_lang="EN-US")
                        content = deepl_client.translate_text(content, target_lang="EN-US")
                    f.write(f"{str(blockID)}\n")
                    f.write(f"{str(title)}\n")
                    f.write(f"{str(content)}\n")
                    f.write("\n")
        except Exception as e:
            error_reports.append(docID)
    return
    
def download_rss_reports(report_feed: str = None, cutoff_date = None, translate: bool = False) -> None:
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
        "macro_feeds": macro_feeds,
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
    filepath = RSS_OUTPUT_PATH / f"rss_feed_output_{datetime.now().strftime('%Y%m%d')}.txt"

    report_feed = parameter_list[report_feed]

    # Download RSS feeds and write to file
    with open(os.path.expanduser(filepath), "w") as file:
        for i in report_feed:
                for name, url in i.items():
                    feed = feedparser.parse(url)
                    file.write(f"Feed: {name.upper()}\n\n")
                    for entry in feed.entries:
                        formatted_time = None
                        if 'published' in entry and entry.published:
                            formatted_time = time.strftime('%Y-%m-%d', entry.published_parsed)
                        if 'updated' in entry and entry.updated:
                            formatted_time = time.strftime('%Y-%m-%d', entry.updated_parsed)
                        if formatted_time and (formatted_time >= cutoff_date):
                            file.write(f"Published: {formatted_time}\n")
                            if 'title' in entry and entry.title:
                                if i == report_feed[1] and translate: # Japanese feeds
                                    title = deepl_client.translate_text(entry.title, target_lang="EN-US").text
                                    file.write(f"Title: {title}\n")
                                else:
                                    file.write(f"Title: {entry.title}\n")
                            if 'summary' in entry and entry.summary:
                                if i == report_feed[1] and translate: # Japanese feeds
                                    summary = deepl_client.translate_text(entry.summary, target_lang="EN-US").text
                                    file.write(f"Summary: {summary}\n")
                                else:
                                    file.write(f"Summary: {entry.summary}\n")
                            if 'link' in entry and entry.link:
                                file.write(f"Link: {entry.link}\n")
                            file.write("\n")
                    file.write("====================================\n\n")
    return

def clear_folder(folder_path: Path):
    for file in folder_path.iterdir():
        if file.is_file():
            file.unlink()

def clear_edinet_reports():
    """Clear all files in the EDINET reports directory."""
    clear_folder(EDINET_REPORTS_PATH)

def clear_rss_reports():
    """Clear all files in the RSS feed output directory."""
    clear_folder(RSS_OUTPUT_PATH)

def vectorize_edinet_reports():
    """Ingest all text files in the EDINET_reports directory into the vector store."""
    ingest_directory(EDINET_REPORTS_PATH)
    clear_edinet_reports()

def vectorize_rss_reports():
    """Ingest all text files in the RSS_feed_output directory into the vector store."""
    ingest_directory(RSS_OUTPUT_PATH)
    clear_rss_reports()

### QUESTION ANSWERING SETUP ###
def answer_question_old(question: str):
    
    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}"
            for doc in docs        )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer the user's question using ONLY the context provided. If the answer is not in the context, say 'I don't know.'"
        ),
        (
            "human",
            """Context:{context} Question:{question}""")])
    # 1. Retrieve
    docs = retriever._get_relevant_documents(question, run_manager=None)

    # 2. Format context
    context = format_docs(docs)

    # 3. Build prompt
    messages = prompt.format_messages(
        context=context,
        question=question
    )

    # 4. Call LLM
    response = llm(messages)
    return response.content

@app.route('/download_edinet_reports', methods=['GET'])
def get_edinet_reports():
    mode = request.args.get("mode", "portfolio")
    ticker = request.args.get("ticker", None)
    translate = request.args.get("translate", False)
    edinet_report_downloader(mode=mode, ticker=ticker, translate=translate)
    return jsonify({"status": "EDINET reports downloaded."})

@app.route('/download_rss_reports', methods=['GET'])
def get_rss_reports():
    report_feed = request.args.get("report_feed", "macro_feeds")
    cutoff_date = request.args.get("earliest_date", "2023-01-01")
    translate = request.args.get("translate", False)
    download_rss_reports(report_feed=report_feed, cutoff_date=cutoff_date, translate=translate)
    return jsonify({"status": "RSS reports downloaded."})

@app.route('/clear_edinet_reports', methods=['GET'])
def clear_edinet_reports_route():
    """Clear all files in the EDINET reports directory."""
    clear_edinet_reports()
    return jsonify({"status": "EDINET reports cleared."})

@app.route('/clear_rss_reports', methods=['GET'])
def clear_rss_reports_route():
    """Clear all files in the RSS feed output directory."""
    clear_rss_reports()
    return jsonify({"status": "RSS reports cleared."})

@app.route('/vectorize_edinet_reports', methods=['GET'])
def vectorize_edinet_reports_route():
    """Ingest all text files in the EDINET_reports directory into the vector store."""
    initial_vector_count = int(index.describe_index_stats().total_vector_count)
    vectorize_edinet_reports()
    updated_vector_count = int(index.describe_index_stats().total_vector_count)
    vectors_added = updated_vector_count - initial_vector_count
    return jsonify({"status": "EDINET reports vectorized.", "vectors_added": vectors_added})

@app.route('/vectorize_rss_reports', methods=['GET'])
def vectorize_rss_reports_route():
    """Ingest all text files in the RSS_feed_output directory into the vector store."""
    initial_vector_count = int(index.describe_index_stats().total_vector_count)
    vectorize_rss_reports()
    updated_vector_count = int(index.describe_index_stats().total_vector_count)
    vectors_added = updated_vector_count - initial_vector_count
    return jsonify({"status": "RSS reports vectorized.", "vectors_added": vectors_added})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}    )

    # 1️⃣ Retrieve relevant documents
    docs = retriever.invoke(user_message)
    context = "\n\n".join(f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs)

    # 2️⃣ Build prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer the user's question using ONLY the context provided. "
            "If the answer is not in the context, say 'I don't know.'"
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

    # 3️⃣ Load conversation history
    history = memory.load_memory_variables({}).get("history", [])

    # 4️⃣ Build messages
    messages = prompt.format_messages(
        context=context,
        question=user_message
    )

    # Insert history between system context and current question
    messages = messages[:2] + history + messages[2:]

    # 5️⃣ Call LLM
    response = llm.invoke(messages)

    # 6️⃣ Save to memory
    memory.save_context(
        {"input": user_message},
        {"output": response.content}
    )

    return jsonify({"response": response.content})


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

if __name__ == '__main__':
    app.run(debug=True)

# === To delete the Pinecone index, uncomment the following lines ===
# pc.delete_index(name=index_name)
# print("Pinecone index deleted.")

# === To view Pinecone index stats, uncomment the following line ===
# print(index.describe_index_stats())