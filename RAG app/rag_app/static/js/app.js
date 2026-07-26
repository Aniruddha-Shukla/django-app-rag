/**
 * NexusRAG Client Application JavaScript
 * Handles API calls, dynamic chat rendering, context inspection, and document ingestion.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const messagesContainer = document.getElementById('messages-container');
    const btnSend = document.getElementById('btn-send');
    const latencyBadge = document.getElementById('latency-metric-badge');

    const ingestTextarea = document.getElementById('ingest-text-input');
    const ingestSourceName = document.getElementById('ingest-source-name');
    const btnIngestText = document.getElementById('btn-ingest-text');
    const btnClearDb = document.getElementById('btn-clear-db');
    const btnSeedData = document.getElementById('btn-seed-data');

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const btnBrowseFile = document.getElementById('btn-browse-file');

    const kbChunkCount = document.getElementById('kb-chunk-count');
    const toggleSidebarBtn = document.getElementById('toggle-sidebar');
    const sidebar = document.getElementById('sidebar');

    // Auto-resize chat textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = (chatInput.scrollHeight) + 'px';
    });

    // Toggle Sidebar
    toggleSidebarBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });

    // Quick Prompt Chips
    document.querySelectorAll('.chip[data-prompt]').forEach(chip => {
        chip.addEventListener('click', () => {
            const promptText = chip.getAttribute('data-prompt');
            chatInput.value = promptText;
            chatInput.focus();
            chatInput.dispatchEvent(new Event('input'));
        });
    });

    // Ingest Seed Data Button
    if (btnSeedData) {
        btnSeedData.addEventListener('click', async () => {
            const seedText = `Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances Large Language Models (LLMs) by retrieving authoritative, up-to-date domain knowledge from external vector databases like ChromaDB before generating a response.

Key Components of RAG:
1. Document Processing: Documents are parsed and split into semantic text chunks with configurable token overlaps to maintain contextual continuity.
2. Vector Embedding: Local neural models (such as TensorFlow or SentenceTransformers) compute dense mathematical vector representations for each text chunk.
3. Vector Database (ChromaDB): ChromaDB indices chunk vectors using approximate nearest neighbor algorithms (e.g. HNSW cosine metric) for sub-millisecond retrieval.
4. Prompt Augmentation: Retrieved top-K matching contexts are injected into the LLM system prompt as grounded reference data.
5. Gemini Generation: Google Gemini 2.5 Flash processes the query alongside the grounded context to generate factual, cited responses without hallucinations.`;

            ingestTextarea.value = seedText;
            ingestSourceName.value = "rag_architecture_overview.md";
            await handleTextIngestion();
        });
    }

    // Submit Chat Form
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        // Clear input
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // Add user message bubble
        appendMessage('user', query);

        // Add loading bot bubble
        const botMessageId = appendLoadingMessage();

        try {
            const startTime = performance.now();
            const response = await fetch('/api/query/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, n_results: 4 })
            });

            const data = await response.json();
            const elapsed = Math.round(performance.now() - startTime);

            if (response.ok) {
                updateBotMessage(botMessageId, data);
                latencyBadge.innerHTML = `<i class="fa-solid fa-stopwatch"></i> RAG Latency: ${data.metrics.total_latency_ms}ms (Retrieval: ${data.metrics.retrieval_latency_ms}ms | Gen: ${data.metrics.generation_latency_ms}ms)`;
            } else {
                updateBotMessage(botMessageId, { answer: `⚠️ Error: ${data.error || 'Failed to process query'}` });
            }

        } catch (err) {
            updateBotMessage(botMessageId, { answer: `⚠️ Connection Error: ${err.message}` });
        }
    });

    // Ingest Text Action
    btnIngestText.addEventListener('click', handleTextIngestion);

    async function handleTextIngestion() {
        const text = ingestTextarea.value.trim();
        const source = ingestSourceName.value.trim() || 'manual_input.txt';

        if (!text) {
            showToast('Please enter text to ingest into Knowledge Base.', 'error');
            return;
        }

        btnIngestText.disabled = true;
        btnIngestText.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Indexing...';

        try {
            const response = await fetch('/api/ingest/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, source_name: source })
            });

            const data = await response.json();
            if (response.ok) {
                showToast(`Success! Ingested ${data.chunks_added} chunks from ${data.source}`, 'success');
                ingestTextarea.value = '';
                refreshStats();
            } else {
                showToast(`Ingestion Failed: ${data.error}`, 'error');
            }
        } catch (err) {
            showToast(`Error: ${err.message}`, 'error');
        } finally {
            btnIngestText.disabled = false;
            btnIngestText.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> Index Text';
        }
    }

    // Drag and Drop File Ingestion
    btnBrowseFile.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) uploadFile(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
    });

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        showToast(`Uploading and chunking ${file.name}...`, 'info');

        try {
            const response = await fetch('/api/ingest/', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (response.ok) {
                showToast(`Indexed ${file.name}: ${data.chunks_added} chunks added to ChromaDB`, 'success');
                refreshStats();
            } else {
                showToast(`Upload Failed: ${data.error}`, 'error');
            }
        } catch (err) {
            showToast(`Upload Error: ${err.message}`, 'error');
        }
    }

    // Clear Database
    btnClearDb.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to clear all vectors from ChromaDB?')) return;

        try {
            const response = await fetch('/api/clear/', { method: 'POST' });
            const data = await response.json();
            if (response.ok) {
                showToast('ChromaDB knowledge base cleared.', 'success');
                refreshStats();
            }
        } catch (err) {
            showToast(`Clear Error: ${err.message}`, 'error');
        }
    });

    // Refresh System Stats
    async function refreshStats() {
        try {
            const res = await fetch('/api/stats/');
            const data = await res.json();
            if (kbChunkCount) kbChunkCount.textContent = data.total_chunks;
        } catch (e) {
            console.error('Failed to fetch stats:', e);
        }
    }

    // Append Message to UI
    function appendMessage(sender, text) {
        const row = document.createElement('div');
        row.className = `message-row ${sender}-row`;

        const avatar = document.createElement('div');
        avatar.className = `avatar ${sender}-avatar`;
        avatar.innerHTML = sender === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = text;

        row.appendChild(avatar);
        row.appendChild(bubble);
        messagesContainer.appendChild(row);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function appendLoadingMessage() {
        const msgId = 'msg-' + Date.now();
        const row = document.createElement('div');
        row.className = 'message-row bot-row';
        row.id = msgId;

        const avatar = document.createElement('div');
        avatar.className = 'avatar bot-avatar';
        avatar.innerHTML = '<i class="fa-solid fa-brain fa-pulse"></i>';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Retrieving vector context & generating response...';

        row.appendChild(avatar);
        row.appendChild(bubble);
        messagesContainer.appendChild(row);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        return msgId;
    }

    function updateBotMessage(msgId, data) {
        const row = document.getElementById(msgId);
        if (!row) return;

        const avatar = row.querySelector('.avatar');
        avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';

        const bubble = row.querySelector('.bubble');
        
        // Format markdown response text basic paragraphs
        let formattedText = data.answer.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
        bubble.innerHTML = formattedText;

        // Render Context Sources Card if sources exist
        if (data.sources && data.sources.length > 0) {
            const sourcesCard = document.createElement('div');
            sourcesCard.className = 'sources-card';

            let sourcesHTML = `<div class="sources-card-header"><i class="fa-solid fa-database"></i> Retained ChromaDB Context Chunks (${data.sources.length})</div>`;
            data.sources.forEach((src, idx) => {
                const sourceName = src.metadata.source || 'Document';
                const score = (src.similarity_score * 100).toFixed(1);
                sourcesHTML += `
                    <div class="source-item">
                        <div class="source-meta">
                            <span><strong>[Source ${idx + 1}]</strong> ${sourceName}</span>
                            <span>Similarity: <strong>${score}%</strong></span>
                        </div>
                        <div class="source-text">"${escapeHtml(src.text.substring(0, 180))}..."</div>
                    </div>
                `;
            });

            sourcesCard.innerHTML = sourcesHTML;
            bubble.appendChild(sourcesCard);
        }

        // Render Latency & Model Pill
        if (data.metrics) {
            const metricsRow = document.createElement('div');
            metricsRow.className = 'metrics-row';
            metricsRow.innerHTML = `
                <span class="metric-pill"><i class="fa-solid fa-microchip"></i> Model: ${data.model}</span>
                <span class="metric-pill"><i class="fa-solid fa-stopwatch"></i> Total: ${data.metrics.total_latency_ms}ms</span>
            `;
            bubble.appendChild(metricsRow);
        }

        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function escapeHtml(text) {
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let icon = '<i class="fa-solid fa-info-circle"></i>';
        if (type === 'success') icon = '<i class="fa-solid fa-circle-check"></i>';
        if (type === 'error') icon = '<i class="fa-solid fa-circle-exclamation"></i>';

        toast.innerHTML = `${icon} <span>${message}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 4000);
    }
});
