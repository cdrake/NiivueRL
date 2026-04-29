import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

// Minimal Markdown → HTML for the notes page. Supports headings, paragraphs,
// inline links/code, and blank-line-separated paragraphs. Avoids pulling in a
// 100KB markdown dependency for a single static page.
function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function inline(s: string): string {
  let out = escapeHtml(s);
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
  return out;
}

function renderMarkdown(md: string): string {
  const lines = md.split('\n');
  const out: string[] = [];
  let para: string[] = [];
  const flush = () => {
    if (para.length) {
      out.push(`<p>${inline(para.join(' '))}</p>`);
      para = [];
    }
  };
  for (const line of lines) {
    if (/^#\s+/.test(line)) {
      flush();
      out.push(`<h1>${inline(line.replace(/^#\s+/, ''))}</h1>`);
    } else if (/^##\s+/.test(line)) {
      flush();
      out.push(`<h2>${inline(line.replace(/^##\s+/, ''))}</h2>`);
    } else if (/^###\s+/.test(line)) {
      flush();
      out.push(`<h3>${inline(line.replace(/^###\s+/, ''))}</h3>`);
    } else if (line.trim() === '') {
      flush();
    } else {
      para.push(line.trim());
    }
  }
  flush();
  return out.join('\n');
}

export default function NotesPage() {
  const [html, setHtml] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const base = import.meta.env.BASE_URL;
    fetch(`${base}notes.md`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((txt) => setHtml(renderMarkdown(txt)))
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <header style={{ padding: '20px 28px', borderBottom: '1px solid #2a2a5a', background: '#16213e', flexShrink: 0 }}>
        <Link to="/" style={{ fontSize: 12, color: '#88f' }}>← back to experiments</Link>
      </header>
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
        <article
          style={{
            maxWidth: 760,
            margin: '0 auto',
            padding: '32px 28px 64px',
            lineHeight: 1.6,
            fontSize: 15,
            color: '#ddd',
          }}
        >
          {error ? (
            <div style={{ color: '#f88' }}>Failed to load notes: {error}</div>
          ) : (
            <div className="notes-md" dangerouslySetInnerHTML={{ __html: html }} />
          )}
        </article>
      </div>
      <style>{`
        .notes-md h1 { color: #4fc3f7; font-size: 24px; margin-bottom: 16px; }
        .notes-md h2 { color: #4fc3f7; font-size: 17px; margin-top: 28px; margin-bottom: 10px; border-bottom: 1px solid #2a2a5a; padding-bottom: 4px; }
        .notes-md h3 { color: #aac; font-size: 14px; margin-top: 20px; margin-bottom: 8px; }
        .notes-md p { margin-bottom: 12px; }
        .notes-md a { color: #88f; }
        .notes-md code { background: #16213e; padding: 1px 5px; border-radius: 3px; font-size: 13px; }
      `}</style>
    </div>
  );
}
