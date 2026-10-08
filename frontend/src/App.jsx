import { useState, useEffect, useCallback, useRef } from 'react';
import { Upload, Search, Trash2, ChevronDown, ChevronRight, Check, X, RefreshCw, Moon, Sun, Copy, Scale } from 'lucide-react';

// API base: explicit VITE_API_BASE > dev localhost > same origin (production
// builds are served by the FastAPI app itself, so origin == backend).
const API =
  import.meta.env.VITE_API_BASE ||
  (import.meta.env.DEV ? 'http://localhost:8000' : window.location.origin);

const TABS = ['Documents', 'Search', 'Review', 'Draft'];

const DOC_TYPES = [
  ['case_file', 'Case file'],
  ['statute', 'Statute'],
  ['judgment', 'Judgment'],
];

const DRAFT_TYPES = [
  ['bail_application', 'Bail application'],
  ['legal_notice', 'Legal notice'],
  ['affidavit', 'Affidavit'],
];

const SEARCH_EXAMPLES = ['accused name', 'date of incident', 'bail application'];

const parseFields = (text) =>
  text.split(',').map((s) => s.trim()).filter(Boolean);

const getInitialTheme = () => {
  // Explicit ?theme=light|dark wins, then saved choice, then system preference
  const param = new URLSearchParams(window.location.search).get('theme');
  if (param === 'dark' || param === 'light') return param;
  try {
    const stored = localStorage.getItem('theme');
    if (stored === 'dark' || stored === 'light') return stored;
  } catch {
    // storage unavailable — fall through to system preference
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

async function postJson(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}

export default function App() {
  const [tab, setTab] = useState('Documents');
  const [health, setHealth] = useState(null);
  const [online, setOnline] = useState(false);
  const [theme, setTheme] = useState(getInitialTheme);
  const [toast, setToast] = useState(null);
  const [copied, setCopied] = useState(false);
  const toastTimer = useRef(null);

  // Documents
  const [docs, setDocs] = useState([]);
  const [docType, setDocType] = useState('case_file');
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState(null); // {ok, text}
  const [expandedId, setExpandedId] = useState(null);
  const [chunksByDoc, setChunksByDoc] = useState({});
  const fileRef = useRef(null);

  // Search
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState(null);
  const [searchErr, setSearchErr] = useState('');

  // Review
  const [reviewDocIds, setReviewDocIds] = useState([]); // empty = all documents
  const [reviewFields, setReviewFields] = useState('accused_name, FIR_number, date_of_incident, medical_condition');
  const [reviewing, setReviewing] = useState(false);
  const [review, setReview] = useState(null);
  const [reviewErr, setReviewErr] = useState('');

  // Draft
  const [draftType, setDraftType] = useState('bail_application');
  const [draftFields, setDraftFields] = useState('accused_name, FIR_number, medical_condition');
  const [drafting, setDrafting] = useState(false);
  const [draft, setDraft] = useState(null);
  const [draftErr, setDraftErr] = useState('');

  // Source side panel
  const [source, setSource] = useState(null);

  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API}/health`);
      if (!res.ok) throw new Error('bad response');
      setHealth(await res.json());
      setOnline(true);
    } catch {
      setHealth(null);
      setOnline(false);
    }
  }, []);

  const fetchDocs = useCallback(async () => {
    try {
      const res = await fetch(`${API}/documents`);
      if (!res.ok) throw new Error('bad response');
      setDocs(await res.json());
      setOnline(true);
    } catch {
      setDocs([]);
    }
  }, []);

  useEffect(() => {
    let active = true;
    const load = async () => {
      // Health + document list are fetched once on mount (external system sync)
      const [hRes, dRes] = await Promise.all([
        fetch(`${API}/health`).catch(() => null),
        fetch(`${API}/documents`).catch(() => null),
      ]);
      if (!active) return;
      if (hRes && hRes.ok) {
        setHealth(await hRes.json());
        setOnline(true);
      } else {
        setHealth(null);
        setOnline(false);
      }
      if (dRes && dRes.ok) {
        setDocs(await dRes.json());
      } else {
        setDocs([]);
      }
    };
    load();
    return () => {
      active = false;
    };
  }, []);

  // Apply + persist the theme
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem('theme', theme);
    } catch {
      // storage unavailable — theme still applies for this session
    }
  }, [theme]);

  // Live status polling; Escape closes the source panel
  useEffect(() => {
    const id = setInterval(fetchHealth, 30000);
    const onKey = (e) => {
      if (e.key === 'Escape') setSource(null);
    };
    window.addEventListener('keydown', onKey);
    return () => {
      clearInterval(id);
      window.removeEventListener('keydown', onKey);
    };
  }, [fetchHealth]);

  useEffect(() => () => clearTimeout(toastTimer.current), []);

  const notify = (text) => {
    setToast(text);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 2500);
  };

  const onPickFile = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadMsg({ ok: false, text: 'Only PDF files are supported.' });
      return;
    }
    setUploading(true);
    setUploadMsg(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('doc_type', docType);
      const res = await fetch(`${API}/documents/upload`, { method: 'POST', body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || 'Upload failed');
      setUploadMsg({
        ok: true,
        text: `Indexed "${data.filename}" — ${data.num_pages} page(s), ${data.num_chunks} chunk(s).`,
      });
      await fetchDocs();
    } catch (err) {
      setUploadMsg({ ok: false, text: err.message || 'Upload failed' });
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const toggleChunks = async (docId) => {
    if (expandedId === docId) {
      setExpandedId(null);
      return;
    }
    setExpandedId(docId);
    if (chunksByDoc[docId]) return;
    try {
      const res = await fetch(`${API}/documents/${docId}/chunks`);
      if (!res.ok) return;
      const data = await res.json();
      setChunksByDoc((prev) => ({ ...prev, [docId]: data.chunks }));
    } catch {
      // backend offline — leave collapsed content empty
    }
  };

  const deleteDoc = async (docId) => {
    const doc = docs.find((d) => d.doc_id === docId);
    const name = doc?.filename ?? docId;
    if (!window.confirm(`Delete "${name}" and all of its chunks?`)) return;
    try {
      const res = await fetch(`${API}/documents/${docId}`, { method: 'DELETE' });
      notify(res.ok ? `Deleted ${name}` : 'Delete failed');
    } catch {
      notify('Delete failed — backend unreachable');
    }
    await fetchDocs();
  };

  const runSearch = async (event, override) => {
    event?.preventDefault();
    const q = (override ?? query).trim();
    if (!q) return;
    if (override) setQuery(override);
    setSearching(true);
    setSearchErr('');
    try {
      const res = await fetch(`${API}/search?q=${encodeURIComponent(q)}&top_k=8`);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || 'Search failed');
      setResults(data.results);
    } catch (err) {
      setResults([]);
      setSearchErr(err.message || 'Search failed');
    } finally {
      setSearching(false);
    }
  };

  const runReview = async () => {
    setReviewing(true);
    setReview(null);
    setReviewErr('');
    try {
      const data = await postJson('/review', {
        doc_ids: reviewDocIds,
        required_fields: parseFields(reviewFields),
      });
      setReview(data);
    } catch (err) {
      setReviewErr(err.message || 'Review failed');
    } finally {
      setReviewing(false);
    }
  };

  const runDraft = async () => {
    setDrafting(true);
    setDraft(null);
    setDraftErr('');
    try {
      const data = await postJson('/draft', {
        case_id: 'local',
        doc_type: draftType,
        required_fields: parseFields(draftFields),
      });
      setDraft(data);
    } catch (err) {
      setDraftErr(err.message || 'Draft failed');
    } finally {
      setDrafting(false);
    }
  };

  // Load a cited chunk and verify the quote against it (Hard Rule #1)
  const copyChunkId = async (chunkId) => {
    try {
      await navigator.clipboard.writeText(chunkId);
      setCopied(true);
      if (toastTimer.current) clearTimeout(toastTimer.current);
      toastTimer.current = setTimeout(() => setCopied(false), 1500);
    } catch {
      notify('Could not copy to clipboard');
    }
  };

  const showSource = async (chunkId, quote) => {
    setSource({ loading: true, chunk_id: chunkId });
    try {
      const srcRes = await fetch(`${API}/source/${chunkId}`);
      if (!srcRes.ok) throw new Error('Source chunk not found in index.');
      const src = await srcRes.json();
      let verify = null;
      if (quote) {
        try {
          verify = await postJson('/verify', { chunk_id: chunkId, quote });
        } catch {
          verify = null;
        }
      }
      setSource({ ...src, verify, quote });
    } catch (err) {
      setSource({ error: err.message || 'Failed to load source', chunk_id: chunkId });
    }
  };

  return (
    <div className="app">
      <div className="tricolor" aria-hidden="true" />
      <header className="app-header">
        <div className="brand">
          <span className="seal" aria-hidden="true">
            <Scale size={22} />
          </span>
          <span>
            <h1 className="brand-title">VERITAS LEGAL</h1>
            <small>Legal Document Intelligence System &middot; Don&apos;t just cite. Prove.</small>
          </span>
        </div>
        <div className="row">
          <div className="status">
            <span className={`dot ${online ? '' : 'err'}`} />
            {online
              ? `${health?.indexed_documents ?? 0} docs · ${health?.indexed_chunks ?? 0} chunks · ${
                  health?.llm_available ? 'LLM mode' : 'grounded demo mode'
                }`
              : 'backend offline'}
          </div>
          <button
            className="btn"
            onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
            title="Toggle light/dark theme"
          >
            {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
          <button className="btn" onClick={() => { fetchHealth(); fetchDocs(); }}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </header>

      <nav className="tabs">
        {TABS.map((name) => (
          <button
            key={name}
            className={`tab ${tab === name ? 'active' : ''}`}
            onClick={() => setTab(name)}
          >
            {name}
          </button>
        ))}
      </nav>

      <main className="main">
        {tab === 'Documents' && (
          <>
            <section className="card">
              <h3>Upload a PDF</h3>
              <div className="row">
                <label className="field-label" htmlFor="doc-type">Type</label>
                <select id="doc-type" className="input" value={docType} onChange={(e) => setDocType(e.target.value)}>
                  {DOC_TYPES.map(([value, label]) => (
                    <option key={value} value={value}>{label}</option>
                  ))}
                </select>
                <input
                  type="file"
                  accept="application/pdf"
                  ref={fileRef}
                  onChange={onPickFile}
                  style={{ display: 'none' }}
                />
                <button className="btn btn-primary" disabled={uploading} onClick={() => fileRef.current?.click()}>
                  <Upload size={14} /> {uploading ? 'Indexing…' : 'Choose PDF'}
                </button>
              </div>
              {uploadMsg && (
                <div className={`msg ${uploadMsg.ok ? 'ok' : 'err'}`}>{uploadMsg.text}</div>
              )}
            </section>

            <section className="card">
              <h3>Indexed documents</h3>
              {docs.length === 0 ? (
                <div className="empty">No documents yet. Upload a PDF to index it.</div>
              ) : (
                <table className="table">
                  <thead>
                    <tr>
                      <th>File</th>
                      <th>Type</th>
                      <th>Pages</th>
                      <th>Chunks</th>
                      <th>Indexed</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {docs.map((doc) => (
                      <DocRow
                        key={doc.doc_id}
                        doc={doc}
                        expanded={expandedId === doc.doc_id}
                        chunks={chunksByDoc[doc.doc_id]}
                        onToggle={() => toggleChunks(doc.doc_id)}
                        onDelete={() => deleteDoc(doc.doc_id)}
                      />
                    ))}
                  </tbody>
                </table>
              )}
            </section>
          </>
        )}

        {tab === 'Search' && (
          <section className="card">
            <h3>Hybrid search (vector + BM25)</h3>
            <form className="row" onSubmit={runSearch}>
              <input
                className="input grow"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g. accused name and date of arrest"
              />
              <button className="btn btn-primary" disabled={searching || !query.trim()}>
                <Search size={14} /> {searching ? 'Searching…' : 'Search'}
              </button>
            </form>
            <div className="row mt" style={{ gap: 6 }}>
              <span className="muted" style={{ fontSize: 12 }}>Try:</span>
              {SEARCH_EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  className="example-chip"
                  disabled={searching}
                  onClick={(e) => runSearch(e, ex)}
                >
                  {ex}
                </button>
              ))}
            </div>
            {searchErr && <div className="msg err">{searchErr}</div>}
            {results && results.length === 0 && !searchErr && (
              <div className="empty">No matching chunks.</div>
            )}
            {results && results.length > 0 && (
              <div className="mt">
                {results.map((r) => (
                  <div className="result" key={r.chunk_id}>
                    <div className="meta">
                      <span>score {r.score.toFixed(3)}</span>
                      <span>{r.doc_id}</span>
                      <span>page {r.page}</span>
                      <button className="btn btn-ghost" onClick={() => showSource(r.chunk_id, null)}>
                        source
                      </button>
                    </div>
                    <div>{r.text.length > 400 ? `${r.text.slice(0, 400)}…` : r.text}</div>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {tab === 'Review' && (
          <>
            <section className="card">
              <h3>Case review</h3>
              <p className="muted" style={{ margin: '0 0 12px' }}>
                Extracts facts with verbatim quotes, then reports missing fields and contradictions
                before anything is drafted.
              </p>
              <div className="row">
                <label className="field-label">Documents</label>
                <select
                  className="input"
                  value={reviewDocIds.length === 1 ? reviewDocIds[0] : ''}
                  onChange={(e) => setReviewDocIds(e.target.value ? [e.target.value] : [])}
                >
                  <option value="">All documents</option>
                  {docs.map((d) => (
                    <option key={d.doc_id} value={d.doc_id}>{d.filename}</option>
                  ))}
                </select>
              </div>
              <div className="row mt">
                <label className="field-label" htmlFor="review-fields">Required fields</label>
                <input
                  id="review-fields"
                  className="input grow"
                  value={reviewFields}
                  onChange={(e) => setReviewFields(e.target.value)}
                />
                <button className="btn btn-primary" disabled={reviewing} onClick={runReview}>
                  {reviewing ? 'Reviewing…' : 'Run review'}
                </button>
              </div>
              {reviewErr && <div className="msg err">{reviewErr}</div>}
            </section>

            {review && (
              <>
                <section className="card">
                  <h3>Extracted facts</h3>
                  {review.extracted_facts.length === 0 ? (
                    <div className="empty">No facts extracted from the selected documents.</div>
                  ) : (
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Field</th>
                          <th>Value</th>
                          <th>Quote</th>
                        </tr>
                      </thead>
                      <tbody>
                        {review.extracted_facts.map((f, i) => (
                          <tr key={`${f.field}-${i}`}>
                            <td className="mono">{f.field}</td>
                            <td>{f.value}</td>
                            <td>
                              {f.chunk_id && f.quote ? (
                                <span className="quote" onClick={() => showSource(f.chunk_id, f.quote)}>
                                  “{f.quote.length > 140 ? `${f.quote.slice(0, 140)}…` : f.quote}”
                                </span>
                              ) : (
                                <span className="muted">—</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </section>

                <section className="card">
                  <h3>Gap report</h3>
                  <div className="section-title">Missing fields</div>
                  {review.gap_report.missing_fields.length === 0 ? (
                    <span className="badge ok">none</span>
                  ) : (
                    <div className="row">
                      {review.gap_report.missing_fields.map((field) => (
                        <span className="chip bad" key={field}>{field}</span>
                      ))}
                    </div>
                  )}

                  <div className="section-title">Contradictions</div>
                  {review.gap_report.contradictions.length === 0 ? (
                    <span className="muted">None detected.</span>
                  ) : (
                    review.gap_report.contradictions.map((c, i) => (
                      <div key={`${c.field}-${i}`} className="mt">
                        <strong className="mono">{c.field}</strong>
                        {c.values.map((v, j) => (
                          <div key={j} className="claim">
                            <div className="body">
                              <div>{v.value}</div>
                              {v.chunk_id && v.quote && (
                                <span className="quote" onClick={() => showSource(v.chunk_id, v.quote)}>
                                  “{v.quote}”
                                </span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    ))
                  )}
                </section>
              </>
            )}
          </>
        )}

        {tab === 'Draft' && (
          <>
            <section className="card">
              <h3>Draft document</h3>
              <div className="row">
                <label className="field-label" htmlFor="draft-type">Document</label>
                <select id="draft-type" className="input" value={draftType} onChange={(e) => setDraftType(e.target.value)}>
                  {DRAFT_TYPES.map(([value, label]) => (
                    <option key={value} value={value}>{label}</option>
                  ))}
                </select>
                <span className="spacer" />
                <button className="btn btn-primary" disabled={drafting} onClick={runDraft}>
                  {drafting ? 'Drafting…' : 'Draft'}
                </button>
              </div>
              <div className="row mt">
                <label className="field-label" htmlFor="draft-fields">Required fields</label>
                <input
                  id="draft-fields"
                  className="input grow"
                  value={draftFields}
                  onChange={(e) => setDraftFields(e.target.value)}
                />
              </div>
              {draftErr && <div className="msg err">{draftErr}</div>}
            </section>

            {draft && (
              <section className="card">
                {draft.sections.length === 0 && (
                  <div className="empty">Nothing drafted — index a document first.</div>
                )}
                {draft.sections.map((section, i) => (
                  <div key={i}>
                    <div className="section-title">{section.heading}</div>
                    {section.claims.map((claim, j) => (
                      <div className="claim" key={j}>
                        {claim.status === 'verified'
                          ? <Check size={15} style={{ color: 'var(--success)', flexShrink: 0, marginTop: 3 }} />
                          : <X size={15} style={{ color: 'var(--danger)', flexShrink: 0, marginTop: 3 }} />}
                        <div className="body">
                          <span className={claim.status === 'verified' ? '' : 'missing'}>
                            {claim.text}
                          </span>
                          {claim.chunk_id && claim.quote && (
                            <div>
                              <span className="quote" onClick={() => showSource(claim.chunk_id, claim.quote)}>
                                cite source
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ))}

                <div className="section-title">Gap report</div>
                <div className="row">
                  {draft.gap_report?.missing_fields?.length > 0 ? (
                    draft.gap_report.missing_fields.map((field) => (
                      <span className="chip bad" key={field}>{field}</span>
                    ))
                  ) : (
                    <span className="badge ok">no missing fields</span>
                  )}
                </div>
              </section>
            )}
          </>
        )}
      </main>

      <footer className="site-footer">
        <span className="footer-title">Veritas Legal &mdash; Agentic Legal Assistant</span>
        <span>
          For demonstration and research purposes only. Not affiliated with any government,
          court, or judiciary.
        </span>
      </footer>

      {source && (
        <aside className="panel">
          <div className="panel-head">
            <h3>Source chunk</h3>
            <div className="row" style={{ gap: 6 }}>
              {!source.loading && !source.error && (
                <button className="btn" onClick={() => copyChunkId(source.chunk_id)}>
                  {copied ? <Check size={14} /> : <Copy size={14} />}
                  {copied ? 'Copied' : 'Copy ID'}
                </button>
              )}
              <button className="btn btn-ghost" onClick={() => setSource(null)} title="Close (Esc)">
                <X size={16} />
              </button>
            </div>
          </div>
          {source.loading ? (
            <div className="muted">Loading…</div>
          ) : source.error ? (
            <div className="msg err">{source.error}</div>
          ) : (
            <>
              <div className="row mono muted" style={{ marginBottom: 10 }}>
                <span>{source.chunk_id}</span>
                <span>page {source.page}</span>
                <span>[{source.char_start}:{source.char_end}]</span>
              </div>
              {source.verify && (
                <div style={{ marginBottom: 10 }}>
                  {source.verify.verified ? (
                    <span className="badge ok">
                      verified · chars {source.verify.char_start}–{source.verify.char_end}
                    </span>
                  ) : (
                    <span className="badge bad">not grounded</span>
                  )}
                  <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>{source.verify.reason}</div>
                </div>
              )}
              <div className="source-text">{source.text}</div>
            </>
          )}
        </aside>
        )}

      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}

function DocRow({ doc, expanded, chunks, onToggle, onDelete }) {
  return (
    <>
      <tr>
        <td className="filename" title={doc.filename}>{doc.filename}</td>
        <td><span className="badge">{doc.doc_type}</span></td>
        <td>{doc.num_pages || '—'}</td>
        <td>{doc.num_chunks}</td>
        <td className="mono muted">
          {doc.uploaded_at ? new Date(doc.uploaded_at).toLocaleString() : '—'}
        </td>
        <td>
          <div className="row" style={{ justifyContent: 'flex-end', gap: 4 }}>
            <button className="btn btn-ghost" onClick={onToggle} title="Chunks">
              {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              chunks
            </button>
            <button className="btn btn-ghost" onClick={onDelete} title="Delete document">
              <Trash2 size={14} />
            </button>
          </div>
        </td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={6} style={{ background: 'var(--surface-2)' }}>
            {!chunks ? (
              <div className="muted">Loading chunks…</div>
            ) : chunks.length === 0 ? (
              <div className="muted">No chunks.</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 260, overflowY: 'auto', padding: '4px 0' }}>
                {chunks.map((chk) => (
                  <div key={chk.chunk_id} className="mono muted">
                    <span style={{ color: 'var(--accent)' }}>{chk.chunk_id}</span>{' '}
                    · page {chk.page} · para {chk.para_id} · [{chk.char_start}:{chk.char_end}]
                    <div style={{ color: 'var(--text)', fontFamily: 'inherit' }}>{chk.text_preview}</div>
                  </div>
                ))}
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}
