document.addEventListener('DOMContentLoaded', () => {
    const uploadedDocIds = [];
    // 1. Drag & Drop Upload Logic
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');

    // Click to select files
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (!files.length) return;
        [...files].forEach(uploadFile);
    }

    async function uploadFile(file) {
        // Create UI element for the uploading file
        const id = Math.random().toString(36).substr(2, 9);
        const itemEl = document.createElement('div');
        itemEl.className = 'upload-item';
        itemEl.id = `upload-${id}`;
        
        itemEl.innerHTML = `
            <div class="upload-item-info">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--text-secondary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                <div>
                    <div class="upload-item-name" title="${file.name}">${file.name}</div>
                    <div class="upload-item-id">Uploading...</div>
                </div>
            </div>
            <div class="spinner"></div>
        `;
        
        uploadStatus.prepend(itemEl);

        const formData = new FormData();
        formData.append('file', file);

        try {
            // API Call
            const response = await fetch('/api/v1/documents/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`Upload failed with status ${response.status}`);
            
            const data = await response.json();
            const docId = data.document_id || data.id || id;
            
            const idText = itemEl.querySelector('.upload-item-id');
            idText.textContent = `Processing & Indexing...`;
            idText.style.color = '#facc15';

            // Poll status until READY or FAILED
            let isReady = false;
            for (let attempt = 0; attempt < 60; attempt++) {
                await new Promise(r => setTimeout(r, 1000));
                try {
                    const statusRes = await fetch(`/api/v1/documents/${docId}/status`);
                    if (statusRes.ok) {
                        const statusData = await statusRes.json();
                        const st = (statusData.status || '').toLowerCase();
                        if (st === 'ready' || st === 'completed') {
                            isReady = true;
                            break;
                        }
                        if (st === 'failed') {
                            throw new Error(statusData.error_message || 'Indexing failed');
                        }
                        idText.textContent = `Status: ${st}...`;
                    }
                } catch (e) {
                    console.warn('Status poll warning:', e);
                }
            }

            // Update UI on completion
            const spinner = itemEl.querySelector('.spinner');
            if (isReady) {
                spinner.outerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4ade80" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
                uploadedDocIds.push(docId);
                idText.textContent = `Ready: ${docId.substring(0, 8)}...`;
                idText.style.color = '#4ade80';
            } else {
                // Timeout or default ready fallback
                spinner.outerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4ade80" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
                uploadedDocIds.push(docId);
                idText.textContent = `Uploaded: ${docId.substring(0, 8)}...`;
                idText.style.color = '#4ade80';
            }

        } catch (error) {
            console.error(error);
            const spinner = itemEl.querySelector('.spinner');
            const idText = itemEl.querySelector('.upload-item-id');
            
            spinner.outerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" x2="9" y1="9" y2="15"/><line x1="9" x2="15" y1="9" y2="15"/></svg>`;
            idText.textContent = 'Upload failed';
            idText.style.color = '#ef4444';
        }
    }

    // 2. Chat Interface Logic
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Add user message
        appendMessage(message, 'user');
        chatInput.value = '';
        
        // Show typing indicator
        const typingEl = document.createElement('div');
        typingEl.className = 'message ai-message';
        typingEl.innerHTML = `
            <div class="avatar ai-avatar">▲</div>
            <div class="message-bubble">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatMessages.appendChild(typingEl);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            // Actual integration endpoint
            const response = await fetch('/api/v1/chat/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: message, document_ids: uploadedDocIds.length > 0 ? uploadedDocIds : null })
            });
            
            if (!response.ok) throw new Error("Chat query failed");
            
            const data = await response.json();
            const reply = data.answer || data.response;
            
            // Remove typing indicator
            typingEl.remove();
            
            // Build rich AI response
            const msgEl = document.createElement('div');
            msgEl.className = 'message ai-message';
            
            // Parse answer text - convert [N] references to citation badges
            let answerHtml = escapeHtml(data.answer || data.response || '');
            answerHtml = answerHtml.replace(/\[(\d+)\]/g, '<span class="citation-badge" data-source="$1" onclick="highlightSource($1)">$1</span>');
            
            // Mode badge
            const modeIcon = data.mode === 'llm' ? '🤖' : '📋';
            const modeLabel = data.mode === 'llm' ? 'LLM Generated' : 'Extractive';
            
            let citationsHtml = '';
            if (data.citations && data.citations.length > 0) {
                citationsHtml = '<div class="sources-container"><div class="sources-header">📎 Sources</div>';
                data.citations.forEach(c => {
                    const docName = c.document_name || 'Document';
                    const pageNum = c.page_number != null ? c.page_number : '—';
                    const relevance = c.relevance_score != null ? (c.relevance_score * 100).toFixed(0) : '—';
                    const section = c.section_title ? `<span class="source-section">§ ${escapeHtml(c.section_title)}</span>` : '';
                    citationsHtml += `
                        <div class="source-card" id="source-${c.rank}">
                            <div class="source-rank">${c.rank}</div>
                            <div class="source-details">
                                <div class="source-doc-name">${escapeHtml(docName)}</div>
                                <div class="source-meta">
                                    <span class="source-page">📄 Page ${pageNum}</span>
                                    ${section}
                                    <span class="source-relevance">Relevance: ${relevance}%</span>
                                </div>
                            </div>
                        </div>`;
                });
                citationsHtml += '</div>';
            }
            
            // Faithfulness meter  
            let faithHtml = '';
            if (data.faithfulness != null && data.faithfulness !== undefined) {
                const pct = (data.faithfulness * 100).toFixed(0);
                const color = data.faithfulness >= 0.8 ? '#4ade80' : data.faithfulness >= 0.5 ? '#facc15' : '#ef4444';
                faithHtml = `
                    <div class="faithfulness-meter">
                        <span class="faith-label">Grounding</span>
                        <div class="faith-bar-bg">
                            <div class="faith-bar-fill" style="width: ${pct}%; background: ${color}"></div>
                        </div>
                        <span class="faith-value" style="color: ${color}">${pct}%</span>
                    </div>`;
            }
            
            msgEl.innerHTML = `
                <div class="avatar ai-avatar">▲</div>
                <div class="message-bubble ai-bubble">
                    <div class="answer-header">
                        <span class="mode-badge">${modeIcon} ${modeLabel}</span>
                    </div>
                    <div class="answer-text">${answerHtml}</div>
                    ${citationsHtml}
                    ${faithHtml}
                </div>
            `;
            
            chatMessages.appendChild(msgEl);
            chatMessages.scrollTop = chatMessages.scrollHeight;

        } catch (error) {
            console.error(error);
            typingEl.remove();
            appendMessage("Sorry, I encountered an error while processing your request.", 'ai');
        }
    });

    function appendMessage(text, sender) {
        const msgEl = document.createElement('div');
        msgEl.className = `message ${sender}-message`;
        
        const avatar = sender === 'user' ? 'U' : '▲';
        
        msgEl.innerHTML = `
            <div class="avatar ${sender}-avatar">${avatar}</div>
            <div class="message-bubble">${escapeHtml(text)}</div>
        `;
        
        chatMessages.appendChild(msgEl);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        let escaped = unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
        
        // Basic markdown formatting
        return escaped
             .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
             .replace(/\*(.*?)\*/g, '<em>$1</em>')
             .replace(/\n/g, '<br>');
    }
});

function highlightSource(num) {
    const el = document.getElementById('source-' + num);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        el.classList.add('source-highlight');
        setTimeout(() => el.classList.remove('source-highlight'), 2000);
    }
}
