/* FOUNDATION 44 — app shell: run orchestration, board, ledger, stats */

Store.load();

/* Link Blank Page and Vault prompts to real corpus atoms where the answer
   text matches, so the daily heartbeat actually moves clause integrity. */
(function link(){
  const byAns = {};
  CORPUS.forEach(a => { if(a.answer) byAns[norm(a.answer)] = a; });
  Object.keys(BLANKPAGE).forEach(c => {
    BLANKPAGE[c].items.forEach(it => { it.atom = byAns[norm(it.answer)] || null; });
  });
  /* Vault rows carry generated atom ids (see corpus-appendix.js). */
  const byId = {};
  CORPUS.forEach(a => { byId[a.id] = a; });
  VAULT.discrete.concat(VAULT.continuous).forEach(r => {
    r.atoms = (r.atomIds || []).map(id => (id && byId[id]) || null);
  });
})();

const $ = s => document.querySelector(s);
const view = () => $('#view');
let RUN = null;

/* ═══ Scoring hook shared by every mode ══════════════════════════════ */
function handleAtom(r){
  const atom = r.atom;
  const id   = atom ? atom.id : r.atomId;
  const type = atom ? atom.type : (r.type || 'FORMULA');
  const yld  = atom ? atom.yield : (r.yield || 2);
  const clause = atom ? atom.clause : (r.clause || null);

  const grade = Sched.grade(r.correct, r.latency, type);
  if(id){ Sched.review(id, grade); }
  if(atom) record(atom, r.correct, r.latency, grade);

  if(RUN){
    if(r.correct) RUN.streak++; else RUN.streak = 0;
    RUN.seen++;
    if(r.correct){
      RUN.hits++;
      const spine = clause ? Board.spine(clause) : 1;
      RUN.points += Sched.points({ type, yield:yld }, r.latency, RUN.streak, spine);
    }
    if(RUN.streak > Store.data.streakBest) Store.data.streakBest = RUN.streak;
    if(!r.correct && atom) RUN.missed.push(atom);
  }
  Store.save();
}

/* ═══ Home ═══════════════════════════════════════════════════════════ */
function home(){
  RUN = null;
  const now = Date.now();
  const v = view(); v.innerHTML = '';

  const overall = Board.overall(now);
  const hero = el('section', 'hero');
  hero.appendChild(el('div', 'eyebrow', 'Foundation 44'));
  hero.appendChild(el('h1', null, 'Mastery, twelve minutes a day.'));

  const ring = el('div', 'ring');
  ring.appendChild(el('div', 'ring-n', overall + ''));
  ring.appendChild(el('div', 'ring-l', 'overall integrity'));
  ring.style.setProperty('--pct', overall);
  hero.appendChild(ring);

  const courses = el('div', 'course-row');
  ['M1', 'S1', 'M2', 'S2', 'AP'].forEach(c => {
    const val = Board.courseIntegrity(c, now);
    const box = el('div', 'course-chip ' + Board.state(val));
    box.appendChild(el('span', 'cc-k', c));
    box.appendChild(el('span', 'cc-v', val + ''));
    const bar = el('div', 'cc-bar'); const f = el('div', 'cc-fill');
    f.style.width = val + '%'; bar.appendChild(f); box.appendChild(bar);
    courses.appendChild(box);
  });
  hero.appendChild(courses);
  v.appendChild(hero);

  /* Damage report */
  const dmg = Board.damage(now, 3);
  const d = el('section', 'panel');
  d.appendChild(el('h2', null, 'Damage report'));
  if(!dmg.length) d.appendChild(el('p', 'muted', 'Everything is solid. Run anyway — decay is continuous.'));
  else {
    const list = el('div', 'dmg-list');
    dmg.forEach(x => {
      const row = el('div', 'dmg ' + Board.state(x.v));
      row.appendChild(el('span', 'dmg-id', x.id));
      row.appendChild(el('span', 'dmg-v', x.v + ''));
      const bar = el('div', 'dmg-bar'); const f = el('div', 'dmg-fill');
      f.style.width = x.v + '%'; bar.appendChild(f); row.appendChild(bar);
      list.appendChild(row);
    });
    d.appendChild(list);
  }
  const go = el('button', 'btn btn-hero', 'Start the run  ▸');
  go.onclick = startRun;
  d.appendChild(go);
  v.appendChild(d);

  /* Free play */
  const fp = el('section', 'panel');
  fp.appendChild(el('h2', null, 'Free play'));
  fp.appendChild(el('p', 'muted', 'Single modes, no run structure. Points still count; the scheduler still learns.'));
  const grid = el('div', 'mode-grid');
  const defs = [
    ['Blank Page', 'Produce it cold, from memory', () => solo(BlankPage, { count:8 })],
    ['Conditions Gate', 'Five seconds. One wrong call ends it.', () => solo(Gate, {})],
    ['Duel', 'Assign properties to confusable siblings', () => solo(Duel, {})],
    ['Chain Builder', 'Put the procedure in order', () => solo(Chain, {})],
    ['The Vault', 'The distribution table, column by column', () => solo(Vault, { count:6 })],
    ['BOSS: Test Selection', Boss.unlocked() ? 'Unlocked — three in a row' : 'Locked: needs S2.03, S2.08, S2.10, S2.12 solid', () => Boss.unlocked() ? solo(Boss, {}) : null]
  ];
  defs.forEach(([n, sub, fn]) => {
    const b = el('button', 'mode' + (n.startsWith('BOSS') && !Boss.unlocked() ? ' mode-locked' : ''));
    b.appendChild(el('span', 'mode-n', n));
    b.appendChild(el('span', 'mode-s', sub));
    b.onclick = fn;
    grid.appendChild(b);
  });
  fp.appendChild(grid);
  v.appendChild(fp);

  const nav = el('div', 'row row-c');
  const bb = el('button', 'btn btn-ghost', 'The board');   bb.onclick = board;
  const sb = el('button', 'btn btn-ghost', 'Error atlas'); sb.onclick = stats;
  nav.appendChild(bb); nav.appendChild(sb);
  v.appendChild(nav);
}

/* ═══ Solo mode wrapper ══════════════════════════════════════════════ */
function solo(mode, opts){
  RUN = { points:0, streak:0, seen:0, hits:0, missed:[], solo:true };
  const v = view(); v.innerHTML = '';
  const bar = el('div', 'runbar');
  bar.appendChild(el('span', 'rb-mode', mode.name));
  const pts = el('span', 'rb-pts', '0'); bar.appendChild(pts);
  const quit = el('button', 'btn btn-ghost btn-xs', 'End'); quit.onclick = home;
  bar.appendChild(quit);
  v.appendChild(bar);
  const mount = el('div', 'stage'); v.appendChild(mount);

  mode.run(Object.assign({
    mount,
    onAtom(r){ handleAtom(r); pts.textContent = RUN.points; },
    onDone(res){ soloEnd(mode, res); }
  }, opts || {}));
}

function soloEnd(mode, res){
  const v = view(); v.innerHTML = '';
  const p = el('section', 'panel');
  p.appendChild(el('h2', null, mode.name + ' complete'));
  const st = el('div', 'stat-row');
  const add = (k, val) => { const b = el('div', 'stat'); b.appendChild(el('div', 'stat-v', String(val))); b.appendChild(el('div', 'stat-k', k)); st.appendChild(b); };
  add('points', RUN.points);
  if(res.streak != null) add('streak', res.streak);
  if(res.total) add('correct', res.correct + '/' + res.total);
  p.appendChild(st);

  if(RUN.missed.length){
    p.appendChild(el('h3', null, 'Repair queue'));
    const list = el('div', 'dmg-list');
    RUN.missed.slice(0, 8).forEach(a => {
      const row = el('div', 'dmg cracked');
      row.appendChild(el('span', 'dmg-id', a.clause));
      row.appendChild(el('span', 'dmg-t', a.prompt || a.claim || a.left));
      list.appendChild(row);
    });
    p.appendChild(list);
    const rp = el('button', 'btn btn-primary', 'Repair these now');
    const missed = RUN.missed.slice();
    rp.onclick = () => repair(missed, home);
    p.appendChild(rp);
  }
  const b = el('button', 'btn btn-ghost', 'Back'); b.onclick = home;
  p.appendChild(b);
  v.appendChild(p);
}

/* ═══ The 12-minute run ══════════════════════════════════════════════ */
function startRun(){
  RUN = { points:0, streak:0, seen:0, hits:0, missed:[], phase:0, t0:Date.now() };
  runShell();
  nextPhase();
}

let RUNBAR = null;
function runShell(){
  const v = view(); v.innerHTML = '';
  const bar = el('div', 'runbar');
  const label = el('span', 'rb-mode', '');
  const pts = el('span', 'rb-pts', '0');
  const quit = el('button', 'btn btn-ghost btn-xs', 'Abandon');
  quit.onclick = () => { if(confirm('Abandon this run? Progress on answered atoms is already saved.')) home(); };
  bar.appendChild(label); bar.appendChild(pts); bar.appendChild(quit);
  v.appendChild(bar);
  const mount = el('div', 'stage'); v.appendChild(mount);
  RUNBAR = { label, pts, mount };
}

function phaseCtx(extra){
  return Object.assign({
    mount:RUNBAR.mount,
    onAtom(r){ handleAtom(r); RUNBAR.pts.textContent = RUN.points; },
    onDone(){ nextPhase(); }
  }, extra || {});
}

function nextPhase(){
  const p = RUN.phase++;
  const set = t => { RUNBAR.label.textContent = t; };

  if(p === 0){                       // Blank Page
    const c = Queue.weakestCourse();
    set('Blank Page · ' + COURSES[c]);
    return BlankPage.run(phaseCtx({ course:c, count:6, onDone(res){ RUN.bpMissed = res.missed || []; nextPhase(); } }));
  }
  if(p === 1){                       // Repair round — not skippable
    const q = RUN.missed.slice();
    if(!q.length){ return nextPhase(); }
    set('Repair round');
    return repair(q, nextPhase, RUNBAR.mount);
  }
  if(p >= 2 && p <= 4){              // Three interleaved draws
    const pool = [Gate, Duel, Chain, Vault];
    const mode = pool[Math.floor(Math.random() * pool.length)];
    set('Interleaved · ' + mode.name);
    return mode.run(phaseCtx(mode === Vault ? { count:4 } : {}));
  }
  if(p === 5){                       // Boss, if earned
    if(Boss.unlocked() && RUN.streak >= 3){
      set('BOSS · Test Selection');
      return Boss.run(phaseCtx({ need:3, onDone(res){
        if(res.defeated){ Store.data.boss.defeated++; Store.data.boss.until = Date.now() + 72 * 3600000; Store.save(); }
        nextPhase();
      }}));
    }
    return nextPhase();
  }
  return ledger();
}

/* Immediate re-test of everything just missed — the highest-yield 90s. */
function repair(queue, done, mount){
  mount = mount || RUNBAR.mount;
  let i = 0;
  const step = () => {
    if(i >= queue.length) return done();
    const atom = queue[i++];
    Card.render(atom, mount, r => {
      handleAtom({ atom, correct:r.correct, latency:r.latency });
      if(RUNBAR) RUNBAR.pts.textContent = RUN.points;
      step();
    });
  };
  step();
}

/* ═══ Ledger ═════════════════════════════════════════════════════════ */
function ledger(){
  const now = Date.now();
  const v = view(); v.innerHTML = '';
  const p = el('section', 'panel');
  p.appendChild(el('div', 'eyebrow', 'Run complete'));
  p.appendChild(el('h2', null, 'Ledger'));

  const st = el('div', 'stat-row');
  const add = (k, val) => { const b = el('div', 'stat'); b.appendChild(el('div', 'stat-v', String(val))); b.appendChild(el('div', 'stat-k', k)); st.appendChild(b); };
  add('points', RUN.points);
  add('answered', RUN.seen);
  add('accuracy', RUN.seen ? Math.round(100 * RUN.hits / RUN.seen) + '%' : '—');
  add('best streak', Store.data.streakBest);
  add('integrity', Board.overall(now));
  p.appendChild(st);

  /* Tomorrow's forced queue: worst retrievability at highest yield */
  const q = CORPUS.map(a => ({ a, r:Sched.R(a.id, now) }))
                  .sort((x, y) => (x.r - y.r) || (y.a.yield - x.a.yield))
                  .slice(0, 6);
  p.appendChild(el('h3', null, 'Tomorrow’s forced queue'));
  const list = el('div', 'dmg-list');
  q.forEach(({ a, r }) => {
    const row = el('div', 'dmg ' + Board.state(Math.round(r * 100)));
    row.appendChild(el('span', 'dmg-id', a.clause));
    row.appendChild(el('span', 'dmg-t', a.prompt || a.claim || (a.left + ' ⟷ ' + a.right)));
    list.appendChild(row);
  });
  p.appendChild(list);

  const row = el('div', 'row row-c');
  const again = el('button', 'btn btn-primary', 'Another run'); again.onclick = startRun;
  const b = el('button', 'btn btn-ghost', 'Home'); b.onclick = home;
  row.appendChild(again); row.appendChild(b);
  p.appendChild(row);
  v.appendChild(p);
  Store.save();
}

/* ═══ The board ══════════════════════════════════════════════════════ */
function board(){
  const now = Date.now();
  const v = view(); v.innerHTML = '';
  const p = el('section', 'panel');
  p.appendChild(el('h2', null, 'The board'));
  p.appendChild(el('p', 'muted', 'Integrity decays continuously. A cracked or dark prerequisite scales down everything downstream — that is the spine multiplier.'));

  ['M1', 'S1', 'M2', 'S2', 'AP'].forEach(c => {
    const cs = Board.clauses().filter(x => x.course === c);
    if(!cs.length) return;
    p.appendChild(el('h3', 'terr', COURSES[c]));
    const grid = el('div', 'node-grid');
    cs.forEach(x => {
      const val = Board.integrity(x.id, now);
      const sp  = Board.spine(x.id, now);
      const n = el('div', 'node ' + Board.state(val));
      n.appendChild(el('div', 'node-id', x.id));
      n.appendChild(el('div', 'node-v', val + ''));
      const bar = el('div', 'node-bar'); const f = el('div', 'node-fill');
      f.style.width = val + '%'; bar.appendChild(f); n.appendChild(bar);
      const req = REQUIRES[x.id];
      if(req) n.appendChild(el('div', 'node-req', '← ' + req.join(', ')));
      if(sp < 1) n.appendChild(el('div', 'node-spine', 'spine ×' + sp.toFixed(2)));
      n.title = Board.atomsOf(x.id).length + ' atoms';
      n.onclick = () => practiceClause(x.id);
      grid.appendChild(n);
    });
    p.appendChild(grid);
  });

  const b = el('button', 'btn btn-ghost', 'Home'); b.onclick = home;
  p.appendChild(b);
  v.appendChild(p);
}

function practiceClause(clause){
  const pool = Board.atomsOf(clause);
  RUN = { points:0, streak:0, seen:0, hits:0, missed:[], solo:true };
  const v = view(); v.innerHTML = '';
  const bar = el('div', 'runbar');
  bar.appendChild(el('span', 'rb-mode', clause));
  const pts = el('span', 'rb-pts', '0'); bar.appendChild(pts);
  const quit = el('button', 'btn btn-ghost btn-xs', 'End'); quit.onclick = board;
  bar.appendChild(quit);
  v.appendChild(bar);
  const mount = el('div', 'stage'); v.appendChild(mount);

  const q = Queue.build(pool, Math.min(pool.length, 8));
  let i = 0;
  const step = () => {
    if(i >= q.length){
      const p = el('section', 'panel');
      p.appendChild(el('h2', null, clause + ' — ' + Board.integrity(clause) + ' integrity'));
      const st = el('div', 'stat-row');
      const add = (k, val) => { const b = el('div', 'stat'); b.appendChild(el('div', 'stat-v', String(val))); b.appendChild(el('div', 'stat-k', k)); st.appendChild(b); };
      add('points', RUN.points); add('correct', RUN.hits + '/' + RUN.seen);
      p.appendChild(st);
      const b = el('button', 'btn btn-ghost', 'Back to the board'); b.onclick = board;
      p.appendChild(b);
      mount.innerHTML = ''; mount.appendChild(p);
      return;
    }
    Card.render(q[i++], mount, r => { handleAtom(r); pts.textContent = RUN.points; step(); });
  };
  step();
}

/* ═══ Error atlas ════════════════════════════════════════════════════ */
function stats(){
  const now = Date.now();
  const v = view(); v.innerHTML = '';
  const p = el('section', 'panel');
  p.appendChild(el('h2', null, 'Error atlas'));

  const log = Store.data.log;
  if(!log.length){
    p.appendChild(el('p', 'muted', 'No attempts logged yet. Play a run and this becomes your revision plan.'));
  } else {
    const byClause = {};
    log.forEach(r => {
      const b = byClause[r.clause] || (byClause[r.clause] = { n:0, miss:0, lat:0 });
      b.n++; if(!r.ok) b.miss++; b.lat += r.lat;
    });
    const rows = Object.keys(byClause).map(c => ({
      c, n:byClause[c].n, miss:byClause[c].miss,
      rate:byClause[c].miss / byClause[c].n,
      lat:byClause[c].lat / byClause[c].n
    })).filter(r => r.n >= 2).sort((a, b) => b.rate - a.rate).slice(0, 12);

    const st = el('div', 'stat-row');
    const add = (k, val) => { const b = el('div', 'stat'); b.appendChild(el('div', 'stat-v', String(val))); b.appendChild(el('div', 'stat-k', k)); st.appendChild(b); };
    add('attempts', log.length);
    add('accuracy', Math.round(100 * log.filter(r => r.ok).length / log.length) + '%');
    add('median latency', (log.map(r => r.lat).sort((a, b) => a - b)[Math.floor(log.length / 2)] || 0).toFixed(1) + 's');
    p.appendChild(st);

    p.appendChild(el('h3', null, 'Worst clauses by miss rate'));
    const list = el('div', 'dmg-list');
    rows.forEach(r => {
      const row = el('div', 'dmg ' + (r.rate > 0.5 ? 'dark' : r.rate > 0.25 ? 'cracked' : 'solid'));
      row.appendChild(el('span', 'dmg-id', r.c));
      row.appendChild(el('span', 'dmg-t', Math.round(r.rate * 100) + '% missed · ' + r.n + ' attempts · ' + r.lat.toFixed(1) + 's avg'));
      list.appendChild(row);
    });
    p.appendChild(list);
  }

  const row = el('div', 'row row-c');
  const dl = el('button', 'btn btn-ghost', 'Export CSV');
  dl.onclick = () => {
    const blob = new Blob([toCSV()], { type:'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'foundation44-log.csv';
    a.click();
  };
  const rs = el('button', 'btn btn-ghost btn-danger', 'Reset all progress');
  rs.onclick = () => { if(confirm('Erase all progress on this device? This cannot be undone.')){ Store.reset(); home(); } };
  const b = el('button', 'btn btn-ghost', 'Home'); b.onclick = home;
  row.appendChild(dl); row.appendChild(b); row.appendChild(rs);
  p.appendChild(row);
  v.appendChild(p);
}

/* ═══ Theme + boot ═══════════════════════════════════════════════════ */
(function theme(){
  const KEY = 'foundation44.theme';
  const set = t => {
    document.documentElement.setAttribute('data-theme', t);
    try{ localStorage.setItem(KEY, t); }catch(e){}
    const btn = $('#themeToggle');
    if(btn) btn.textContent = t === 'dark' ? '🌙' : '☀️';
  };
  let saved = null;
  try{ saved = localStorage.getItem(KEY); }catch(e){}
  const start = saved || (window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.addEventListener('DOMContentLoaded', () => {
    set(start);
    $('#themeToggle').onclick = () => set(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
    $('#homeLink').onclick = e => { e.preventDefault(); home(); };
    home();
  });
})();
