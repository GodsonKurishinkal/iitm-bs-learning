/* FOUNDATION 44 — the six modes */

const el = (tag, cls, txt) => {
  const n = document.createElement(tag);
  if(cls) n.className = cls;
  if(txt != null) n.textContent = txt;
  return n;
};
const shuffle = a => { a = a.slice(); for(let i = a.length - 1; i > 0; i--){ const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

/* ═══ Single-atom card renderer ══════════════════════════════════════
   cb({ correct, latency, atom }) fires once the atom is fully answered. */
const Card = {
  render(atom, mount, cb){
    mount.innerHTML = '';
    const t0 = performance.now();
    const done = ok => cb({ correct:ok, latency:(performance.now() - t0) / 1000, atom });

    const head = el('div', 'card-head');
    head.appendChild(el('span', 'card-tag', atom.type));
    head.appendChild(el('span', 'card-clause', atom.clause));
    mount.appendChild(head);

    const fn = this[atom.type];
    if(fn) fn.call(this, atom, mount, done);
    else done(false);
  },

  /* Feedback banner shared by every type */
  feedback(mount, ok, text, next){
    const box = el('div', 'fb ' + (ok ? 'fb-ok' : 'fb-no'));
    box.appendChild(el('strong', null, ok ? 'Correct' : 'Missed'));
    if(text) box.appendChild(el('p', null, text));
    const btn = el('button', 'btn btn-primary', 'Continue');
    btn.onclick = next;
    box.appendChild(btn);
    mount.appendChild(box);
    btn.focus();
  },

  CLASSIFY(atom, mount, done){
    mount.appendChild(el('h3', 'q', atom.prompt));
    const wrap = el('div', 'opts');
    const order = shuffle(atom.options.map((o, i) => ({ o, i })));
    order.forEach(({ o, i }) => {
      const b = el('button', 'opt', o);
      b.onclick = () => {
        const ok = i === atom.answer;
        wrap.querySelectorAll('.opt').forEach(x => x.disabled = true);
        b.classList.add(ok ? 'opt-ok' : 'opt-no');
        if(!ok) Array.from(wrap.children).forEach((c, k) => { if(order[k].i === atom.answer) c.classList.add('opt-ok'); });
        this.feedback(mount, ok, ok ? '' : 'Answer: ' + atom.options[atom.answer], () => done(ok));
      };
      wrap.appendChild(b);
    });
    mount.appendChild(wrap);
  },

  FORMULA(atom, mount, done){ this._typed(atom, mount, done); },
  DERIVE(atom, mount, done){ this._typed(atom, mount, done); },

  _typed(atom, mount, done){
    mount.appendChild(el('h3', 'q', atom.prompt));
    const inp = el('input', 'entry');
    inp.type = 'text'; inp.autocomplete = 'off'; inp.spellcheck = false;
    inp.placeholder = 'type it from memory — plain text is fine';
    mount.appendChild(inp);
    const row = el('div', 'row');
    const go = el('button', 'btn btn-primary', 'Check');
    row.appendChild(go); mount.appendChild(row);

    const submit = () => {
      const ok = matches(inp.value, atom);
      inp.disabled = true; go.disabled = true;
      const box = el('div', 'fb ' + (ok ? 'fb-ok' : 'fb-no'));
      box.appendChild(el('strong', null, ok ? 'Correct' : 'Missed'));
      box.appendChild(el('p', 'ans', atom.answer));
      const r = el('div', 'row');
      const next = el('button', 'btn btn-primary', 'Continue');
      next.onclick = () => done(ok);
      r.appendChild(next);
      if(!ok){
        const ovr = el('button', 'btn btn-ghost', 'I had this right');
        ovr.title = 'Counts as a slow correct — use only when the checker was being pedantic about notation.';
        ovr.onclick = () => done(true);
        r.appendChild(ovr);
      }
      box.appendChild(r);
      mount.appendChild(box);
      next.focus();
    };
    go.onclick = submit;
    inp.onkeydown = e => { if(e.key === 'Enter') submit(); };
    inp.focus();
  },

  CONDITION(atom, mount, done){
    this._verdict(atom, mount, done, atom.prompt, ['accept', 'reject'],
      ['Accept — the tool applies', 'Reject — a condition fails'], atom.verdict);
  },
  TRAP(atom, mount, done){
    this._verdict(atom, mount, done, atom.claim, [true, false],
      ['True', 'False'], atom.verdict);
  },

  /* Two-step: call the verdict, then name the reason. Both must be right. */
  _verdict(atom, mount, done, text, vals, labels, truth){
    mount.appendChild(el('h3', 'q', text));
    const wrap = el('div', 'opts opts-2');
    vals.forEach((v, i) => {
      const b = el('button', 'opt', labels[i]);
      b.onclick = () => {
        const vOK = v === truth;
        wrap.querySelectorAll('.opt').forEach(x => x.disabled = true);
        b.classList.add(vOK ? 'opt-ok' : 'opt-no');
        this._reason(atom, mount, vOK, done);
      };
      wrap.appendChild(b);
    });
    mount.appendChild(wrap);
  },

  _reason(atom, mount, verdictOK, done){
    const rs = atom.reasons || [];
    if(!rs.length) return this.feedback(mount, verdictOK, '', () => done(verdictOK));
    mount.appendChild(el('p', 'sub-q', 'Now name the reason:'));
    const wrap = el('div', 'opts');
    shuffle(rs).forEach(r => {
      const b = el('button', 'opt opt-sm', r.label);
      b.onclick = () => {
        const rOK = r.k === atom.because;
        const ok = verdictOK && rOK;
        wrap.querySelectorAll('.opt').forEach(x => x.disabled = true);
        b.classList.add(rOK ? 'opt-ok' : 'opt-no');
        const right = rs.find(x => x.k === atom.because);
        this.feedback(mount, ok, ok ? '' : 'Because ' + (right ? right.label : ''), () => done(ok));
      };
      wrap.appendChild(b);
    });
    mount.appendChild(wrap);
  },

  DISCRIM(atom, mount, done){
    mount.appendChild(el('h3', 'q', 'Assign each property to the correct side'));
    const duel = el('div', 'duel');
    const L = el('div', 'duel-side'), R = el('div', 'duel-side');
    L.appendChild(el('div', 'duel-name', atom.left));
    R.appendChild(el('div', 'duel-name', atom.right));
    duel.appendChild(L); duel.appendChild(R);
    mount.appendChild(duel);

    const slot = el('div', 'duel-item');
    mount.appendChild(slot);
    const row = el('div', 'row row-c');
    const bl = el('button', 'btn btn-side', '◀  ' + atom.left);
    const br = el('button', 'btn btn-side', atom.right + '  ▶');
    row.appendChild(bl); row.appendChild(br);
    mount.appendChild(row);

    const items = shuffle(atom.items);
    let i = 0, wrong = 0;
    const show = () => {
      if(i >= items.length){
        bl.disabled = br.disabled = true;
        const ok = wrong === 0;
        this.feedback(mount, ok, ok ? 'Clean sweep.' : wrong + ' misassigned.', () => done(ok));
        return;
      }
      slot.textContent = items[i].t;
      slot.className = 'duel-item';
    };
    const answer = side => {
      if(i >= items.length) return;
      const ok = items[i].side === side;
      if(!ok){ wrong++; slot.classList.add('duel-no'); }
      const tgt = side === 'L' ? L : R;
      const chip = el('div', 'chip ' + (ok ? 'chip-ok' : 'chip-no'), items[i].t);
      tgt.appendChild(chip);
      i++;
      setTimeout(show, ok ? 90 : 320);
    };
    bl.onclick = () => answer('L');
    br.onclick = () => answer('R');
    show();
  },

  CHAIN(atom, mount, done){
    mount.appendChild(el('h3', 'q', atom.prompt));
    mount.appendChild(el('p', 'sub-q', 'Click the steps in the correct order.'));
    const pool = el('div', 'chain-pool');
    const built = el('div', 'chain-built');
    mount.appendChild(built); mount.appendChild(pool);

    let next = 0, wrong = 0;
    const order = shuffle(atom.steps.map((s, i) => ({ s, i })));
    order.forEach(({ s, i }) => {
      const b = el('button', 'step', s);
      b.onclick = () => {
        const ok = i === next;
        if(ok){
          b.disabled = true; b.classList.add('step-used');
          const line = el('div', 'chain-line');
          line.appendChild(el('span', 'chain-n', String(next + 1)));
          line.appendChild(el('span', null, s));
          built.appendChild(line);
          next++;
          if(next === atom.steps.length){
            pool.querySelectorAll('.step').forEach(x => x.disabled = true);
            const good = wrong === 0;
            this.feedback(mount, good, good ? 'Correct order.' : wrong + ' out-of-order attempts.', () => done(good));
          }
        } else {
          wrong++;
          b.classList.add('step-no');
          setTimeout(() => b.classList.remove('step-no'), 320);
        }
      };
      pool.appendChild(b);
    });
  }
};

/* ═══ Mode: Blank Page ═══════════════════════════════════════════════ */
const BlankPage = {
  name:'Blank Page',
  run(ctx){
    const course = ctx.course || Queue.weakestCourse();
    const deck = BLANKPAGE[course];
    const items = shuffle(deck.items).slice(0, ctx.count || 6);
    const m = ctx.mount;
    let i = 0, hits = 0; const missed = [];

    const step = () => {
      if(i >= items.length){
        ctx.onDone({ correct:hits, total:items.length, missed, label:deck.title });
        return;
      }
      const it = items[i];
      m.innerHTML = '';
      const head = el('div', 'card-head');
      head.appendChild(el('span', 'card-tag', 'BLANK PAGE'));
      head.appendChild(el('span', 'card-clause', deck.title + ' · ' + (i + 1) + '/' + items.length));
      m.appendChild(head);
      m.appendChild(el('h3', 'q', it.cue));

      const inp = el('input', 'entry');
      inp.type = 'text'; inp.autocomplete = 'off'; inp.spellcheck = false;
      inp.placeholder = 'produce it from memory';
      m.appendChild(inp);
      const row = el('div', 'row');
      const go = el('button', 'btn btn-primary', 'Check');
      row.appendChild(go); m.appendChild(row);

      const t0 = performance.now();
      const submit = () => {
        const ok = matches(inp.value, { answer:it.answer, accept:it.accept });
        const lat = (performance.now() - t0) / 1000;
        inp.disabled = true; go.disabled = true;
        const box = el('div', 'fb ' + (ok ? 'fb-ok' : 'fb-no'));
        box.appendChild(el('strong', null, ok ? 'Correct' : 'Missed'));
        box.appendChild(el('p', 'ans', it.answer));
        const r = el('div', 'row');
        const nx = el('button', 'btn btn-primary', 'Continue');
        const advance = good => {
          if(good) hits++; else missed.push(it);
          /* it.atom links the cue back to its corpus atom, so the daily
             heartbeat moves real clause integrity rather than nothing. */
          ctx.onAtom && ctx.onAtom(it.atom
            ? { correct:good, latency:lat, atom:it.atom }
            : { correct:good, latency:lat, type:'FORMULA', yield:3, course:course });
          i++; step();
        };
        nx.onclick = () => advance(ok);
        r.appendChild(nx);
        if(!ok){
          const ovr = el('button', 'btn btn-ghost', 'I had this right');
          ovr.onclick = () => advance(true);
          r.appendChild(ovr);
        }
        box.appendChild(r); m.appendChild(box); nx.focus();
      };
      go.onclick = submit;
      inp.onkeydown = e => { if(e.key === 'Enter') submit(); };
      inp.focus();
    };
    step();
  }
};

/* ═══ Mode: Conditions Gate — sudden death ═══════════════════════════ */
const Gate = {
  name:'Conditions Gate',
  run(ctx){
    const pool = CORPUS.filter(a => a.type === 'CONDITION' || a.type === 'TRAP');
    const queue = Queue.build(pool, 40);
    const m = ctx.mount;
    let streak = 0, i = 0, timer = null;
    const LIMIT = 5.0;

    const step = () => {
      if(i >= queue.length){ return end(true); }
      const atom = queue[i++];
      m.innerHTML = '';
      const head = el('div', 'card-head');
      head.appendChild(el('span', 'card-tag card-tag-hot', 'GATE'));
      head.appendChild(el('span', 'card-clause', 'streak ' + streak));
      m.appendChild(head);

      const bar = el('div', 'timer'); const fill = el('div', 'timer-fill');
      bar.appendChild(fill); m.appendChild(bar);

      const isCond = atom.type === 'CONDITION';
      m.appendChild(el('h3', 'q', isCond ? atom.prompt : atom.claim));
      const wrap = el('div', 'opts opts-2');
      const truth = isCond ? atom.verdict : atom.verdict;
      const vals  = isCond ? ['accept', 'reject'] : [true, false];
      const labs  = isCond ? ['Accept', 'Reject'] : ['True', 'False'];

      const t0 = performance.now();
      let settled = false;
      const settle = (ok, why) => {
        if(settled) return; settled = true;
        clearInterval(timer);
        const lat = (performance.now() - t0) / 1000;
        ctx.onAtom && ctx.onAtom({ correct:ok, latency:lat, atom });
        wrap.querySelectorAll('.opt').forEach(x => x.disabled = true);
        if(ok){ streak++; setTimeout(step, 260); }
        else {
          const right = (atom.reasons || []).find(x => x.k === atom.because);
          const box = el('div', 'fb fb-no');
          box.appendChild(el('strong', null, why || 'Wrong call'));
          box.appendChild(el('p', null, 'Because ' + (right ? right.label : '—')));
          const b = el('button', 'btn btn-primary', 'End run');
          b.onclick = () => end(false);
          box.appendChild(b); m.appendChild(box); b.focus();
        }
      };

      vals.forEach((v, k) => {
        const b = el('button', 'opt', labs[k]);
        b.onclick = () => {
          const ok = v === truth;
          b.classList.add(ok ? 'opt-ok' : 'opt-no');
          settle(ok);
        };
        wrap.appendChild(b);
      });
      m.appendChild(wrap);

      clearInterval(timer);
      timer = setInterval(() => {
        const e = (performance.now() - t0) / 1000;
        fill.style.width = Math.max(0, 100 - (e / LIMIT) * 100) + '%';
        if(e >= LIMIT) settle(false, 'Out of time');
      }, 50);
    };

    const end = survived => {
      clearInterval(timer);
      ctx.onDone({ streak, survived, correct:streak, total:streak + (survived ? 0 : 1) });
    };
    step();
  }
};

/* ═══ Mode: Duel ═════════════════════════════════════════════════════ */
const Duel = {
  name:'Duel',
  run(ctx){
    const pool = CORPUS.filter(a => a.type === 'DISCRIM');
    const atom = Queue.build(pool, 1)[0];
    Card.render(atom, ctx.mount, r => {
      ctx.onAtom && ctx.onAtom(r);
      ctx.onDone({ correct:r.correct ? 1 : 0, total:1 });
    });
  }
};

/* ═══ Mode: Chain Builder ════════════════════════════════════════════ */
const Chain = {
  name:'Chain Builder',
  run(ctx){
    const pool = CORPUS.filter(a => a.type === 'CHAIN');
    const atom = Queue.build(pool, 1)[0];
    Card.render(atom, ctx.mount, r => {
      ctx.onAtom && ctx.onAtom(r);
      ctx.onDone({ correct:r.correct ? 1 : 0, total:1 });
    });
  }
};

/* ═══ Mode: The Vault — A.01 under time ══════════════════════════════ */
const Vault = {
  name:'The Vault',
  run(ctx){
    const m = ctx.mount;
    const sweep = ctx.sweep || (Math.random() < 0.5 ? 1 : 2); // 1=Mean 2=Variance
    const colName = sweep === 1 ? 'Mean' : 'Variance';
    const colIdx  = sweep === 1 ? 1 : 2;
    const rows = shuffle(VAULT.discrete.concat(VAULT.continuous)).slice(0, ctx.count || 5);
    let i = 0, hits = 0;

    const step = () => {
      if(i >= rows.length){ ctx.onDone({ correct:hits, total:rows.length }); return; }
      const r = rows[i];
      m.innerHTML = '';
      const head = el('div', 'card-head');
      head.appendChild(el('span', 'card-tag', 'VAULT'));
      head.appendChild(el('span', 'card-clause', 'column sweep · ' + colName + ' · ' + (i + 1) + '/' + rows.length));
      m.appendChild(head);
      m.appendChild(el('h3', 'q', r.name));
      m.appendChild(el('p', 'sub-q', 'Give the ' + colName.toLowerCase() + '.'));

      const inp = el('input', 'entry');
      inp.type = 'text'; inp.autocomplete = 'off'; inp.spellcheck = false;
      m.appendChild(inp);
      const row = el('div', 'row');
      const go = el('button', 'btn btn-primary', 'Check');
      row.appendChild(go); m.appendChild(row);

      const t0 = performance.now();
      const submit = () => {
        const target = r.cells[colIdx];
        const ok = matches(inp.value, { answer:target });
        const lat = (performance.now() - t0) / 1000;
        inp.disabled = true; go.disabled = true;
        const linked = r.atoms && r.atoms[colIdx];
        ctx.onAtom && ctx.onAtom(linked
          ? { correct:ok, latency:lat, atom:linked }
          : { correct:ok, latency:lat, type:'FORMULA', yield:3, clause:'A.01' });
        const box = el('div', 'fb ' + (ok ? 'fb-ok' : 'fb-no'));
        box.appendChild(el('strong', null, ok ? 'Correct' : 'Missed'));
        box.appendChild(el('p', 'ans', target));
        const rr = el('div', 'row');
        const nx = el('button', 'btn btn-primary', 'Continue');
        const advance = good => { if(good) hits++; i++; step(); };
        nx.onclick = () => advance(ok);
        rr.appendChild(nx);
        if(!ok){
          const ovr = el('button', 'btn btn-ghost', 'I had this right');
          ovr.onclick = () => advance(true);
          rr.appendChild(ovr);
        }
        box.appendChild(rr); m.appendChild(box); nx.focus();
      };
      go.onclick = submit;
      inp.onkeydown = e => { if(e.key === 'Enter') submit(); };
      inp.focus();
    };
    step();
  }
};

/* ═══ Mode: BOSS — Test Selection ════════════════════════════════════ */
const Boss = {
  name:'Test Selection',
  unlocked(){
    return ['S2.03', 'S2.08', 'S2.10', 'S2.12'].every(c => Board.integrity(c) >= 80);
  },
  scenario(){
    const row = BOSS[Math.floor(Math.random() * BOSS.length)];
    const n1 = 8 + Math.floor(Math.random() * 40), n2 = 8 + Math.floor(Math.random() * 40);
    return { row, n1, n2 };
  },
  run(ctx){
    const m = ctx.mount;
    let inRow = 0, need = ctx.need || 3, attempts = 0;

    const step = () => {
      if(inRow >= need){ ctx.onDone({ defeated:true, correct:inRow, total:attempts }); return; }
      const sc = this.scenario();
      const row = sc.row;
      attempts++;
      m.innerHTML = '';
      const head = el('div', 'card-head');
      head.appendChild(el('span', 'card-tag card-tag-boss', 'BOSS'));
      head.appendChild(el('span', 'card-clause', inRow + '/' + need + ' in a row'));
      m.appendChild(head);
      m.appendChild(el('h3', 'q', row.q));
      m.appendChild(el('p', 'sub-q', 'Conditions: ' + row.cond + ' · n₁ = ' + sc.n1 + ', n₂ = ' + sc.n2));

      const t0 = performance.now();
      const parts = [
        { label:'Which test?',        key:'test', opts:BOSS.map(r => r.test) },
        { label:'Null distribution?', key:'dist', opts:BOSS.map(r => r.dist) },
        { label:'Degrees of freedom?',key:'df',   opts:BOSS.map(r => r.df) }
      ];
      let p = 0, allOK = true;

      const ask = () => {
        if(p >= parts.length){
          const lat = (performance.now() - t0) / 1000;
          ctx.onAtom && ctx.onAtom({ correct:allOK, latency:lat, pseudo:true, clause:'A.02' });
          if(allOK){
            inRow++;
            const box = el('div', 'fb fb-ok');
            box.appendChild(el('strong', null, 'All four parts correct'));
            box.appendChild(el('p', 'ans', row.stat));
            const b = el('button', 'btn btn-primary', inRow >= need ? 'Finish' : 'Next scenario');
            b.onclick = step; box.appendChild(b); m.appendChild(box); b.focus();
          } else {
            const box = el('div', 'fb fb-no');
            box.appendChild(el('strong', null, 'All-or-nothing — streak reset'));
            box.appendChild(el('p', null, row.test + ' · ' + row.dist + ' · df ' + row.df));
            box.appendChild(el('p', 'ans', row.stat));
            const b = el('button', 'btn btn-primary', 'End attempt');
            b.onclick = () => ctx.onDone({ defeated:false, correct:inRow, total:attempts });
            box.appendChild(b); m.appendChild(box); b.focus();
          }
          return;
        }
        const part = parts[p];
        const box = el('div', 'boss-part');
        box.appendChild(el('p', 'sub-q', part.label));
        const wrap = el('div', 'opts');
        const truth = row[part.key];
        const distract = shuffle(part.opts.filter(o => o !== truth)).slice(0, 3);
        shuffle(distract.concat([truth])).forEach(o => {
          const b = el('button', 'opt opt-sm', o);
          b.onclick = () => {
            const ok = o === truth;
            if(!ok) allOK = false;
            wrap.querySelectorAll('.opt').forEach(x => x.disabled = true);
            b.classList.add(ok ? 'opt-ok' : 'opt-no');
            if(!ok) Array.from(wrap.children).forEach(c => { if(c.textContent === truth) c.classList.add('opt-ok'); });
            p++; setTimeout(ask, ok ? 200 : 700);
          };
          wrap.appendChild(b);
        });
        box.appendChild(wrap);
        m.appendChild(box);
      };
      ask();
    };
    step();
  }
};

const MODES = { BlankPage, Gate, Duel, Chain, Vault, Boss };
