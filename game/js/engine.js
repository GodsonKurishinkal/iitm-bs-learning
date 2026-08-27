/* FOUNDATION 44 — engine: storage, scheduler, board math, answer checking */

/* ═══ Answer normalisation ═══════════════════════════════════════════
   Typed answers are compared after aggressive normalisation. This is not a
   computer-algebra system: it folds notation, not mathematics. The UI always
   offers an "I had this right" override for the cases it judges too harshly. */
const SYM = {
  '√':'sqrt','²':'^2','³':'^3','·':'*','×':'*','−':'-','–':'-','—':'-',
  '≤':'<=','≥':'>=','≠':'!=','∞':'inf','∈':'in','⊆':'subset','∅':'empty',
  'α':'alpha','β':'beta','γ':'gamma','δ':'delta','ε':'eps','θ':'theta',
  'λ':'lambda','μ':'mu','ν':'nu','ρ':'rho','σ':'sigma','τ':'tau','φ':'phi',
  'χ':'chi','ψ':'psi','ω':'omega','Γ':'Gamma','Δ':'Delta','Σ':'sum','Π':'prod',
  'Φ':'Phi','Ω':'Omega','∑':'sum','∏':'prod','∫':'int','∂':'d','∇':'grad',
  '∩':'&','∪':'|','⊥':'perp','⟨':'<','⟩':'>','‖':'||','→':'->','⇒':'=>',
  '±':'+-','…':'...','x̄':'xbar','ȳ':'ybar','d̄':'dbar','θ̂':'thetahat',
  'p̂':'phat','λ̂':'lambdahat','μ̂':'muhat','σ̂':'sigmahat','x̂':'xhat',
  '₀':'0','₁':'1','₂':'2','₃':'3','₄':'4','ₙ':'n','ᵢ':'i','ⱼ':'j','ₖ':'k',
  '⁻':'-','ᵀ':'^T','′':'\'','″':'\'\''
};

function norm(s){
  if(s == null) return '';
  let t = String(s);
  for(const k in SYM) t = t.split(k).join(SYM[k]);
  t = t.toLowerCase();
  // LaTeX-ish cleanup
  t = t.replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, '($1)/($2)');
  t = t.replace(/\\d?frac/g, '').replace(/\\[a-z]+/g, m => m.slice(1));
  t = t.replace(/[{}$\\]/g, '');
  // notation folding
  t = t.replace(/\bnorm\s*\(/g, '||(').replace(/\bmathbb/g, '');
  t = t.replace(/\bp\s*\(\s*([a-z])\s*\|\s*([a-z])\s*\)/g, 'p($1|$2)');
  t = t.replace(/\b(?:cov|covariance)\b/g, 'cov').replace(/\b(?:var|variance)\b/g, 'var');
  t = t.replace(/\band\b/g, '&').replace(/\bintersect(?:ion)?\b/g, '&');
  t = t.replace(/\bunion\b/g, '|').replace(/\bimplies\b/g, '=>');
  t = t.replace(/\bsqrt\b/g, 'sqrt').replace(/\broot\b/g, 'sqrt');
  t = t.replace(/\*\*/g, '^');
  // drop cosmetic characters entirely
  t = t.replace(/[\s,;'"`_]/g, '');
  t = t.replace(/[.]+$/, '');
  return t;
}

/* A tolerant comparison: exact after normalisation, or equal once every
   non-alphanumeric character is also stripped (catches bracket-style drift). */
function loose(s){ return norm(s).replace(/[^a-z0-9^]/g, ''); }
function matches(input, atom){
  const cands = [atom.answer].concat(atom.accept || []);
  const a = norm(input), b = loose(input);
  if(!a) return false;
  return cands.some(c => norm(c) === a || loose(c) === b);
}

/* ═══ Persistent state ═══════════════════════════════════════════════ */
const KEY = 'foundation44.v1';
const Store = {
  data:null,
  load(){
    try{ this.data = JSON.parse(localStorage.getItem(KEY)) || null; }
    catch(e){ this.data = null; }
    if(!this.data) this.data = { atoms:{}, log:[], streakBest:0, boss:{ defeated:0, until:0 }, created:Date.now() };
    if(!this.data.atoms) this.data.atoms = {};
    if(!this.data.log) this.data.log = [];
    if(!this.data.boss) this.data.boss = { defeated:0, until:0 };
    return this.data;
  },
  save(){
    try{ localStorage.setItem(KEY, JSON.stringify(this.data)); }
    catch(e){ /* quota or private mode — the run still works in memory */ }
  },
  atom(id){
    if(!this.data.atoms[id]) this.data.atoms[id] = { S:0, t0:0, reps:0, lapses:0 };
    return this.data.atoms[id];
  },
  reset(){ this.data = null; try{ localStorage.removeItem(KEY); }catch(e){} return this.load(); }
};

/* ═══ FSRS-lite scheduler ════════════════════════════════════════════ */
const DAY = 86400000;
const TARGET = { FORMULA:8, CLASSIFY:6, DISCRIM:5, CONDITION:5, CHAIN:25, TRAP:7, DERIVE:60 };
const WEIGHT = { FORMULA:3, CONDITION:3, DISCRIM:2, CLASSIFY:2, TRAP:2, CHAIN:1, DERIVE:1 };
const BASE   = { FORMULA:100, CONDITION:100, DISCRIM:80, CLASSIFY:70, TRAP:80, CHAIN:120, DERIVE:150 };

const Sched = {
  /* Power-law retrievability. Unseen atoms sit at 0. */
  R(id, now){
    const a = Store.atom(id);
    if(!a.reps || !a.S) return 0;
    const t = Math.max(0, ((now || Date.now()) - a.t0) / DAY);
    return 1 / (1 + t / (9 * a.S));
  },
  grade(correct, latencySec, type){
    if(!correct) return 'AGAIN';
    const tgt = TARGET[type] || 10;
    if(latencySec <= tgt) return 'EASY';
    if(latencySec <= 2.5 * tgt) return 'GOOD';
    return 'HARD';
  },
  review(id, grade, now){
    now = now || Date.now();
    const a = Store.atom(id);
    const R = this.R(id, now);
    const S = a.S || 1;
    let S2;
    if(grade === 'AGAIN'){ S2 = Math.max(0.5, S * 0.35); a.lapses++; }
    else if(grade === 'HARD') S2 = S * (1 + 0.15 * R);
    else if(grade === 'GOOD') S2 = S * (1 + 1.1 * (1 - R));
    else                      S2 = S * (1 + 1.9 * (1 - R));
    if(!a.reps) S2 = grade === 'AGAIN' ? 0.5 : (grade === 'EASY' ? 3 : 1.5);
    a.S = Math.min(S2, 365);
    a.t0 = now;
    a.reps++;
    return a;
  },
  points(atom, latencySec, streakCount, spine){
    const tgt = TARGET[atom.type] || 10;
    const speed  = Math.max(0.5, Math.min(2.0, (2 * tgt) / Math.max(latencySec, 0.1)));
    const streak = 1 + 0.1 * Math.min(streakCount, 10);
    return Math.round((BASE[atom.type] || 80) * atom.yield * speed * streak * spine);
  }
};

/* ═══ The board: clauses, dependencies, integrity ════════════════════ */
const REQUIRES = {
  'S1.05':['M1.02'], 'S1.06':['S1.05'], 'S1.07':['S1.06'],
  'M2.08':['M1.04','M2.07'], 'M2.06':['M2.02'], 'M2.07':['M2.06'], 'M2.10':['M2.08','M2.09'],
  'M2.11':['M2.10'], 'M1.06':['S1.04'], 'S1.04':['S1.03'],
  'S2.04':['S2.03'], 'S2.07':['S2.04'], 'S2.08':['S2.07','S2.05'], 'S2.09':['S2.08'],
  'S2.10':['S2.09'], 'S2.12':['S2.10'], 'S2.13':['S2.12','A.02'], 'S2.11':['S1.07'],
  'S2.03':['A.01'], 'S2.05':['A.01']
};
const COURSES = { M1:'Mathematics I', S1:'Statistics I', M2:'Mathematics II', S2:'Statistics II', AP:'Appendix' };

const Board = {
  clauses(){
    const seen = {}, out = [];
    CORPUS.forEach(a => { if(!seen[a.clause]){ seen[a.clause] = 1; out.push({ id:a.clause, course:a.course }); } });
    return out.sort((x, y) => x.id.localeCompare(y.id));
  },
  atomsOf(clause){ return CORPUS.filter(a => a.clause === clause); },
  integrity(clause, now){
    const as = this.atomsOf(clause);
    if(!as.length) return 0;
    let num = 0, den = 0;
    as.forEach(a => {
      const w = WEIGHT[a.type] || 1;
      num += w * Sched.R(a.id, now);
      den += w;
    });
    return Math.round(100 * num / den);
  },
  state(v){ return v >= 80 ? 'solid' : (v >= 40 ? 'cracked' : 'dark'); },
  /* Nearest three ancestors, breadth-first, so the far end of S2 stays playable. */
  ancestors(clause, depth){
    const out = [], seen = {};
    let frontier = REQUIRES[clause] || [];
    for(let d = 0; d < (depth || 3) && frontier.length; d++){
      const next = [];
      frontier.forEach(c => {
        if(seen[c]) return;
        seen[c] = 1; out.push(c);
        (REQUIRES[c] || []).forEach(p => next.push(p));
      });
      frontier = next;
    }
    return out.slice(0, 3);
  },
  /* Has this clause ever been studied? An untouched prerequisite is unvisited,
     not soft — it must not drag its descendants down before you have begun. */
  touched(clause){
    return this.atomsOf(clause).some(a => Store.atom(a.id).reps > 0);
  },
  spine(clause, now){
    const anc = this.ancestors(clause);
    if(!anc.length) return 1;
    let m = 1;
    anc.forEach(a => {
      if(!this.touched(a)) return;          // neutral until you have started it
      const v = this.integrity(a, now) / 80;
      m *= Math.max(0.25, Math.min(1, v));
    });
    return m;
  },
  courseIntegrity(course, now){
    const cs = this.clauses().filter(c => c.course === course);
    if(!cs.length) return 0;
    return Math.round(cs.reduce((s, c) => s + this.integrity(c.id, now), 0) / cs.length);
  },
  overall(now){
    const cs = this.clauses();
    return Math.round(cs.reduce((s, c) => s + this.integrity(c.id, now), 0) / cs.length);
  },
  /* Clauses whose integrity dropped most since the last recorded visit. */
  damage(now, limit){
    return this.clauses()
      .map(c => ({ id:c.id, course:c.course, v:this.integrity(c.id, now) }))
      .filter(c => c.v < 80)
      .sort((a, b) => a.v - b.v)
      .slice(0, limit || 3);
  }
};

/* ═══ Atom selection ═════════════════════════════════════════════════
   Weakest-first, weighted by yield, never two consecutive atoms from the
   same clause (the anti-blocked-practice rule from the design doc). */
const Queue = {
  score(a, now){
    const R = Sched.R(a.id, now);
    const urgency = 1 - R;                    // low retrievability ⇒ high urgency
    const unseen  = Store.atom(a.id).reps ? 0 : 0.35;
    return (urgency + unseen) * (0.6 + 0.4 * a.yield);
  },
  build(pool, n, now){
    now = now || Date.now();
    const ranked = pool.slice().map(a => ({ a, s:this.score(a, now) + Math.random() * 0.18 }))
                        .sort((x, y) => y.s - x.s).map(x => x.a);
    const out = [];
    let lastClause = null;
    // greedy pass honouring the no-two-in-a-row rule
    while(out.length < n && ranked.length){
      let i = ranked.findIndex(a => a.clause !== lastClause);
      if(i < 0) i = 0;
      const pick = ranked.splice(i, 1)[0];
      out.push(pick); lastClause = pick.clause;
    }
    return out;
  },
  byTypes(types, n, now){
    return this.build(CORPUS.filter(a => types.indexOf(a.type) >= 0), n, now);
  },
  byCourse(course, n, now){
    return this.build(CORPUS.filter(a => a.course === course), n, now);
  },
  weakestCourse(now){
    const cs = Object.keys(COURSES).filter(c => c !== 'AP');
    return cs.map(c => ({ c, v:Board.courseIntegrity(c, now) })).sort((a, b) => a.v - b.v)[0].c;
  }
};

/* ═══ Attempt log — the personal error atlas ═════════════════════════ */
function record(atom, correct, latencySec, grade){
  Store.data.log.push({
    t:Date.now(), id:atom.id, clause:atom.clause, course:atom.course,
    type:atom.type, ok:correct ? 1 : 0, lat:Math.round(latencySec * 10) / 10, g:grade
  });
  if(Store.data.log.length > 4000) Store.data.log.splice(0, Store.data.log.length - 4000);
}

function toCSV(){
  const rows = [['timestamp','atom','clause','course','type','correct','latency_s','grade']];
  Store.data.log.forEach(r => rows.push([
    new Date(r.t).toISOString(), r.id, r.clause, r.course, r.type, r.ok, r.lat, r.g
  ]));
  return rows.map(r => r.join(',')).join('\n');
}
