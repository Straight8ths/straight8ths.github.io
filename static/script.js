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

    fetch(url, { method: 'GET' })
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
    fetch(`/vectorize_rss_reports`, { method: 'GET' })
        .then(response => response.json())
        .then(data => {
            addLog('Vectorizing collected news...');
            addLog('News vectorization complete');
            addLog(`Vectors added: ${data.vectors_added}`);
        })
        .catch(error => {
            addLog('ERROR: Failed to vectorize news');
        });
}

function download_edinet_reports() {
    const mode = document.getElementById('filingsMode').value;
    addLog(`Collecting EDINET filings for selected mode: ${mode}`);

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
    fetch(url, { method: 'GET' })
        .then(response => response.json())
        .then(data => {
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
    addLog('Starting vectorization of collected reports...');
    fetch(`/vectorize_edinet_reports`, { method: 'GET' })
        .then(response => response.json())
        .then(data => {
            addLog('Report vectorization complete');
            addLog(`Vectors added: ${data.vectors_added}`);
        })
        .catch(error => {
            addLog('ERROR: Failed to vectorize reports');
        });
}

function appendMessage(role, text) {
  const chat = document.getElementById("chatWindow");

  // Create the message container
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message");

  // Assign user or AI styling
  if (role === "user") {
    messageDiv.classList.add("user-message");
  } else {
    messageDiv.classList.add("ai-message");
  }

  // Set the message text
  messageDiv.innerText = text;

  // Append to chat window
  chat.appendChild(messageDiv);

  // Scroll to the bottom
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById("chatInput");
  const message = input.value.trim();
  if (!message) return;

  appendMessage("user", message);
  input.value = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });
    addLog(`User: ${message}`);

    const data = await response.json();
    appendMessage("ai", data.response);
    addLog(`AI: ${data.response}`);
  } catch (err) {
    appendMessage("ai", "Error: Could not reach server.");
    console.error(err);
  }
}

// Enable Enter key to send
document.getElementById("chatInput").addEventListener("keypress", function(event) {
  if (event.key === "Enter") {
    event.preventDefault();
    sendMessage();
  }
});

function checkVectorDBStatus() {
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

function confirmClearVectorDB() {
    const confirmed = confirm(
        "WARNING\n\n" +
        "This will permanently delete ALL vectors from the database.\n\n" +
        "This action CANNOT be undone.\n\n" +
        "Are you sure you want to continue?"
    );

    if (!confirmed) {
        addLog("Vector DB clear cancelled by user.");
        return;
    }

    clearVectorDB();
}

function clearVectorDB() {
    addLog("Clearing Vector DB...");

    fetch("/clear_vector_db", { method: "POST" })
        .then(response => response.json())
        .then(data => {
            addLog(data.status);
        })
        .catch(error => {
            addLog(`ERROR: Failed to clear Vector DB - ${error}`);
        });
}

async function upload_whitepapers() {
  const input = document.getElementById("whitepaperInput");
  const files = input.files;

  if (!files.length) {
    alert("Select at least one file");
    return;
  }

  const formData = new FormData();
    addLog(`Uploading and vectorizing whitepapers...`);

  for (const file of files) {
    formData.append("whitepapers", file);
  }

  const response = await fetch("/vectorize_whitepapers", {
    method: "POST",
    body: formData
    });

    const data = await response.json();

  if (response.status !== 200) {
    addLog(`ERROR: ${data.error}`);
    return;
  }
  if (data.vectors_added === 0) {
    addLog(`No new vectors were added. Files may have already been vectorized.`);
    return;
  }
    addLog(`Document vectorization complete`);
    addLog(`Vectors added: ${data.vectors_added}`);
    addLog(`Index stats: ${JSON.stringify(data.index_stats)}`);
}

const input = document.getElementById("chatInput");

input.addEventListener("keypress", function(event) {
  if (event.key === "Enter") {
    event.preventDefault(); // prevent form submission / new line
    sendMessage();
  }
});

addLog('System initialized');