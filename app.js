import { h, applyDiff } from 'webjsx';

const REPO = 'https://github.com/AnEntrypoint/streamtts';

const sideSections = [
  {
    heading: 'Project',
    items: [
      { label: 'Overview', href: '#overview', active: true },
      { label: 'Install', href: '#install' },
      { label: 'Architecture', href: '#architecture' },
      { label: 'Changelog', href: '#changelog' },
    ],
  },
  {
    heading: 'Reference',
    items: [
      { label: 'train subcommand', href: '#train' },
      { label: 'inspect subcommand', href: '#inspect' },
      { label: 'merge-stats', href: '#merge-stats' },
      { label: 'Checkpoint format', href: '#checkpoint' },
    ],
  },
  {
    heading: 'Links',
    items: [
      { label: 'GitHub', href: REPO },
      { label: 'ccsniff (npm)', href: 'https://www.npmjs.com/package/ccsniff' },
      { label: 'RWKV-7 HF model', href: 'https://huggingface.co/RWKV/RWKV7-Goose-World3-1.5B-HF' },
      { label: 'AnEntrypoint org', href: 'https://github.com/AnEntrypoint' },
    ],
  },
];

const receipt = [
  { k: 'Status',    v: 'active' },
  { k: 'Language',  v: 'Rust (2021 edition)' },
  { k: 'License',   v: 'MIT' },
  { k: 'Model',     v: 'RWKV-7 1.5B (state-tuned)' },
  { k: 'Tokenizer', v: 'Online bigram-merge (dynamic)' },
  { k: 'Optimizer', v: 'AdamW on trainable subset' },
  { k: 'Memory',    v: '~4 GB (BF16 weights + state)' },
  { k: 'First commit', v: '2026-05-01' },
  { k: 'Crates',    v: 'sttx-core · sttx-ccsniff · sttx-train · sttx-cli' },
];

const changelog = [
  {
    ver: '0.3.0',
    date: '2026-05-01',
    notes: [
      'add train_corpus example + Trainer::step_on_ids for raw-id streaming',
      'end-to-end train witness via synthetic safetensors',
      'document candle test-witness pattern + RWKV-7 tensor layout in AGENTS.md',
    ],
  },
  {
    ver: '0.2.0',
    date: '2026-04-30',
    notes: [
      'state-tuning loop with AdamW on per-layer att_kv prefix tensors',
      'surprise-prioritized replay buffer with configurable capacity',
      'online bigram-merge DynamicTokenizer with Hypernetwork',
      'structured JSONL observability via sttx-core::obs',
    ],
  },
  {
    ver: '0.1.0',
    date: '2026-04-28',
    notes: [
      'initial Rust workspace: sttx-core, sttx-ccsniff, sttx-train, sttx-cli',
      'tokio-based ccsniff subprocess adapter with bounded backpressure',
      'HF model load via hf-hub + candle rwkv_v7::Model',
      'checkpoint serialization with safetensors round-trip',
    ],
  },
];

const archRows = [
  { crate: 'sttx-core',     role: 'Observability, dynamic tokenizer, replay buffer, trace types' },
  { crate: 'sttx-ccsniff',  role: 'Tokio subprocess adapter → typed Trace events from ccsniff JSONL' },
  { crate: 'sttx-train',    role: 'RWKV-7 model load, state-tuning training loop, checkpoint I/O' },
  { crate: 'sttx-cli',      role: 'streamtts binary — clap subcommand dispatch (train / inspect / merge-stats)' },
];

function Topbar() {
  return h('div', { class: 'app-topbar' },
    h('span', { class: 'app-brand' }, 'AnEntrypoint'),
    h('span', { class: 'app-brand-sep' }, '/'),
    h('span', { class: 'app-brand-project' }, 'streamtts'),
    h('a', { class: 'btn-stamp', href: REPO, target: '_blank', rel: 'noopener' }, 'GitHub →'),
  );
}

function Crumb() {
  return h('nav', { class: 'app-crumb' },
    h('a', { href: 'https://github.com/AnEntrypoint' }, 'AnEntrypoint'),
    h('span', null, '/'),
    h('span', null, 'streamtts'),
  );
}

function Side() {
  return h('aside', { class: 'app-side' },
    ...sideSections.map(sec =>
      h('div', { class: 'side-group' },
        h('div', { class: 'side-heading' }, sec.heading),
        ...sec.items.map(it =>
          h('a', { class: 'side-item' + (it.active ? ' active' : ''), href: it.href }, it.label),
        ),
      ),
    ),
  );
}

function Overview() {
  return h('section', { id: 'overview', class: 'panel' },
    h('h1', null, 'streamtts'),
    h('p', { class: 'lede' },
      'RWKV-7 streaming trainer — ingests Claude Code traces via ',
      h('a', { href: 'https://www.npmjs.com/package/ccsniff', target: '_blank' }, 'ccsniff'),
      ', adapts a 1.5B-parameter model in real time using state-tuning + a hypernetwork over an online dynamic tokenizer. Single Rust binary, distributable via ',
      h('code', null, 'cargo build --release'),
      '.',
    ),
    h('div', { class: 'chip-row' },
      h('span', { class: 'chip' }, 'Rust'),
      h('span', { class: 'chip' }, 'RWKV-7'),
      h('span', { class: 'chip' }, 'state-tuning'),
      h('span', { class: 'chip' }, 'online tokenizer'),
      h('span', { class: 'chip' }, 'ccsniff'),
      h('span', { class: 'chip' }, 'candle'),
    ),
  );
}

function Install() {
  return h('section', { id: 'install', class: 'panel' },
    h('h2', null, 'Install'),
    h('p', null, 'Requires Rust 1.77+. No pre-built binaries yet — build from source:'),
    h('div', { class: 'cli' },
      h('pre', null,
        'git clone https://github.com/AnEntrypoint/streamtts\ncd streamtts\ncargo build --release\n# binary: target/release/streamtts',
      ),
    ),
    h('p', null, 'Also install ccsniff to feed Claude Code traces:'),
    h('div', { class: 'cli' },
      h('pre', null, 'npm install -g ccsniff'),
    ),
  );
}

function Receipt() {
  return h('section', { id: 'receipt', class: 'panel' },
    h('h2', null, 'Project metadata'),
    h('div', { class: 'kv-table' },
      ...receipt.map(r =>
        h('div', { class: 'row' },
          h('span', { class: 'kv-key' }, r.k),
          h('span', { class: 'kv-val' }, r.v),
        ),
      ),
    ),
  );
}

function Architecture() {
  return h('section', { id: 'architecture', class: 'panel' },
    h('h2', null, 'Architecture'),
    h('p', null,
      'Four-crate Rust workspace. Training loop runs state-tuning only — frozen RWKV-7 weights with trainable per-layer att_kv prefix tensors and a small hypernetwork. Memory target: 6 GB.',
    ),
    h('div', { class: 'kv-table' },
      ...archRows.map(r =>
        h('div', { class: 'row' },
          h('span', { class: 'kv-key mono' }, r.crate),
          h('span', { class: 'kv-val' }, r.role),
        ),
      ),
    ),
    h('h3', null, 'Memory budget'),
    h('div', { class: 'kv-table' },
      h('div', { class: 'row' }, h('span', { class: 'kv-key' }, 'RWKV-7 BF16 weights'), h('span', { class: 'kv-val' }, '~3 GB')),
      h('div', { class: 'row' }, h('span', { class: 'kv-key' }, 'Activations (ctx ≤ 1024)'), h('span', { class: 'kv-val' }, '~0.5 GB')),
      h('div', { class: 'row' }, h('span', { class: 'kv-key' }, 'Trainable state prefix'), h('span', { class: 'kv-val' }, '~10 MB')),
      h('div', { class: 'row' }, h('span', { class: 'kv-key' }, 'Hypernetwork weights'), h('span', { class: 'kv-val' }, '~10 MB')),
      h('div', { class: 'row' }, h('span', { class: 'kv-key' }, 'Replay buffer (1000×1024)'), h('span', { class: 'kv-val' }, '~4 MB')),
      h('div', { class: 'row' }, h('span', { class: 'kv-key' }, 'AdamW moments'), h('span', { class: 'kv-val' }, '~40 MB')),
    ),
  );
}

function Changelog() {
  return h('section', { id: 'changelog', class: 'panel' },
    h('h2', null, 'Changelog'),
    ...changelog.map(entry =>
      h('div', { class: 'changelog-entry' },
        h('div', { class: 'row' },
          h('span', { class: 'chip' }, 'v' + entry.ver),
          h('span', { class: 'changelog-date' }, entry.date),
        ),
        h('ul', null,
          ...entry.notes.map(n => h('li', null, n)),
        ),
      ),
    ),
  );
}

function Status() {
  return h('div', { class: 'app-status' },
    h('span', null, 'streamtts · RWKV-7 streaming trainer · MIT · '),
    h('a', { href: REPO, target: '_blank', rel: 'noopener' }, 'AnEntrypoint/streamtts'),
  );
}

function App() {
  return h('div', { class: 'app' },
    Topbar(),
    Crumb(),
    h('div', { class: 'app-body' },
      Side(),
      h('main', { class: 'app-main narrow' },
        Overview(),
        Install(),
        Receipt(),
        Architecture(),
        Changelog(),
      ),
    ),
    Status(),
  );
}

applyDiff(document.getElementById('root'), App());
