const API_BASE =
  window.location.hostname === "localhost"
    ? "http://127.0.0.1:5000"
    : "https://your-app.onrender.com";

const dropdown = document.getElementById('filingsMode');
const tickerbox = document.getElementById('tickerbox');
const dateBox = document.getElementById('dateBox');

// Values that should enable the text box
const enableValues = ['name_and_comps'];

dropdown.addEventListener('change', function() {
    if (enableValues.includes(this.value)) {
        tickerbox.disabled = false;
        tickerbox.focus();
    } else {
        tickerbox.disabled = true;
        tickerbox.value = '';
    }
});

function addLog(message) {
    const logsBox = document.getElementById('logsBox');
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.textContent = `[${timestamp}] ${message}`;
    logsBox.appendChild(logEntry);
    logsBox.scrollTop = logsBox.scrollHeight;
}

function download_rss_reports() {
    const report_feed = document.getElementById('newsFeeds').value;
    if (!report_feed) {
        addLog('ERROR: No news feed selected');
        return;
    }
    let url = `/download_rss_reports?report_feed=${report_feed}`;
    
    const earliest_date = document.getElementById('dateBox').value;
    if (earliest_date) {
        url += `&earliest_date=${earliest_date}`;
    }

    const translate = document.getElementById('newstranslateCheckbox').checked;
    if (translate) {
        url += `&translate=true`;
    }

    fetch(url, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            addLog(`Collecting news from: ${report_feed}`);
            addLog(`News collection complete for ${report_feed}`);
        })
        .catch(error => {
            addLog('ERROR: Failed to collect news');
        });
    // "Add to Vector DB" button becomes clickable
    document.querySelector('button[onclick="vectorize_rss_reports()"]').disabled = false;
}

function vectorize_rss_reports() {
    // Starts out disabled
    if (document.querySelector('button[onclick="vectorize_rss_reports()"]').disabled) {
        addLog('ERROR: No news collected to vectorize');
        return;
    }
    fetch(`/vectorize_rss_reports`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            addLog('Vectorizing collected news...');
            addLog('News vectorization complete');
        })
        .catch(error => {
            addLog('ERROR: Failed to vectorize news');
        });
}

function download_edinet_reports() {
    const mode = document.getElementById('filingsMode').value;
    const ticker = document.getElementById('tickerbox').value;
    const translate = document.getElementById('filingstranslateCheckbox').checked;
    let url = `/download_edinet_reports?mode=${mode}`;
    
    if (!mode) {
        addLog('ERROR: No report mode selected');
        return;
    }
    if (mode === 'name_and_comps') {
        if (!ticker) {
            addLog('ERROR: No ticker provided for single name & competitors mode');
            return;
        }
        url += `&ticker=${ticker}`;
    }
    if (mode === 'portfolio') {
        addLog('Collecting filings for portfolio mode...');
    } else {
        addLog(`Collecting filings for ticker: ${ticker} and its competitors...`);
    }
    if (translate) {
        url += `&translate=true`;
    }
    fetch(url, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            addLog(`Collecting filings in mode: ${mode}`);
            addLog(`Filings collection complete for mode: ${mode}`);
        })
        .catch(error => {
            addLog('ERROR: Failed to collect filings');
        });
    // "Add to Vector DB" button becomes clickable
    document.querySelector('button[onclick="vectorize_edinet_reports()"]').disabled = false;
}

function vectorize_edinet_reports() {
    if (document.querySelector('button[onclick="vectorize_edinet_reports()"]').disabled) {
        addLog('ERROR: No reports collected to vectorize');
        return;
    }
    fetch(`/vectorize_edinet_reports`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            addLog('Vectorizing collected reports...');
            addLog('Report vectorization complete');
        })
        .catch(error => {
            addLog('ERROR: Failed to vectorize reports');
        });
}

function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;

    const chatWindow = document.getElementById('chatWindow');
    
    const userMsg = document.createElement('div');
    userMsg.className = 'message user-message';
    userMsg.textContent = message;
    chatWindow.appendChild(userMsg);
    
    addLog(`User sent message: "${message}"`);
    
    input.value = '';
    
    setTimeout(() => {
        const aiMsg = document.createElement('div');
        aiMsg.className = 'message ai-message';
        aiMsg.textContent = 'This is a simulated AI response. Connect to an actual AI API for real responses.';
        chatWindow.appendChild(aiMsg);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        addLog('AI response generated');
    }, 500);
    
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function checkVectorDBHealth() {
    addLog('VectorDB health check initiated...');

    fetch("/vector_db_status", { method: 'GET' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'healthy') {
                addLog('VectorDB is healthy and operational.');
                addLog(`Index stats: ${JSON.stringify(data)}`);
            } else if (data.status === 'failed') {
                addLog(`VectorDB health check failed: ${data.error}`);
            } else {
                addLog('VectorDB returned unexpected response.');
            }
        })
        .catch(error => {
            addLog(`ERROR: Failed to check VectorDB health - ${error}`);
        });
    return;
}

document.getElementById('chatInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

addLog('System initialized');