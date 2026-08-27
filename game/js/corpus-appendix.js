/* FOUNDATION 44 — Corpus: Appendix (A.01 … A.04)
   Also defines the structured decks that power Vault, Boss and Blank Page. */

/* ── A.01 The distribution table — the Vault deck ──────────────────── */
window.VAULT = {
  columns:['PMF / PDF','Mean','Variance','MGF'],
  discrete:[
    { name:'Bernoulli(p)',    cells:['p^x(1-p)^(1-x)','p','p(1-p)','1-p+pe^t'],                models:'single trial' },
    { name:'Binomial(n,p)',   cells:['C(n,x)p^x(1-p)^(n-x)','np','np(1-p)','(1-p+pe^t)^n'],    models:'successes in n fixed trials' },
    { name:'Geometric(p)',    cells:['(1-p)^(x-1)p','1/p','(1-p)/p^2','pe^t/(1-(1-p)e^t)'],    models:'trials until first success' },
    { name:'Poisson(λ)',      cells:['e^(-λ)λ^x/x!','λ','λ','e^(λ(e^t-1))'],                   models:'counts in fixed time or space' },
    { name:'Discrete uniform',cells:['1/n','(n+1)/2','(n^2-1)/12','—'],                        models:'equally likely outcomes' },
    { name:'Hypergeometric',  cells:['C(K,x)C(N-K,n-x)/C(N,n)','nK/N','nK/N·(N-K)/N·(N-n)/(N-1)','—'], models:'sampling without replacement' }
  ],
  continuous:[
    { name:'Uniform(a,b)',    cells:['1/(b-a)','(a+b)/2','(b-a)^2/12','(e^(tb)-e^(ta))/(t(b-a))'] },
    { name:'Exponential(λ)',  cells:['λe^(-λx)','1/λ','1/λ^2','λ/(λ-t)'] },
    { name:'Normal(μ,σ²)',    cells:['(1/(σ√(2π)))e^(-(x-μ)²/2σ²)','μ','σ^2','e^(μt+σ²t²/2)'] },
    { name:'Gamma(α,β)',      cells:['β^α x^(α-1) e^(-βx)/Γ(α)','α/β','α/β^2','(β/(β-t))^α'] },
    { name:'Beta(α,β)',       cells:['x^(α-1)(1-x)^(β-1)/B(α,β)','α/(α+β)','αβ/((α+β)²(α+β+1))','—'] },
    { name:'χ²_k',            cells:['Gamma(k/2, 1/2)','k','2k','(1-2t)^(-k/2)'] },
    { name:'t_k',             cells:['symmetric, heavy-tailed','0  (k>1)','k/(k-2)  (k>2)','does not exist'] }
  ]
};

/* ── A.02 The test selection map — the Boss deck ───────────────────── */
window.BOSS = [
  { q:'Is the mean equal to μ₀?', cond:'σ known',
    test:'One-sample z', stat:'z = (x̄ − μ₀)/(σ/√n)', dist:'N(0,1)', df:'—' },
  { q:'Is the mean equal to μ₀?', cond:'σ unknown',
    test:'One-sample t', stat:'t = (x̄ − μ₀)/(s/√n)', dist:'t_{n−1}', df:'n − 1' },
  { q:'Do two independent groups differ in mean?', cond:'equal variances',
    test:'Pooled t', stat:'t = (x̄₁ − x̄₂ − δ₀)/(s_p√(1/n₁+1/n₂))', dist:'t_{n₁+n₂−2}', df:'n₁ + n₂ − 2' },
  { q:'Do two independent groups differ in mean?', cond:'unequal variances',
    test:'Welch t', stat:'t = (x̄₁ − x̄₂ − δ₀)/√(s₁²/n₁ + s₂²/n₂)', dist:'t, approximate df', df:'approximate' },
  { q:'Did the same units change?', cond:'paired data',
    test:'Paired t', stat:'t = (d̄ − δ₀)/(s_d/√n)', dist:'t_{n−1}', df:'n − 1' },
  { q:'Is a proportion equal to p₀?', cond:'large n',
    test:'z for a proportion', stat:'z = (p̂ − p₀)/√(p₀(1−p₀)/n)', dist:'N(0,1)', df:'—' },
  { q:'Is the variance equal to σ₀²?', cond:'normal population',
    test:'Chi-square for one variance', stat:'χ² = (n−1)s²/σ₀²', dist:'χ²_{n−1}', df:'n − 1' },
  { q:'Are two variances equal?', cond:'normal populations',
    test:'F test', stat:'F = s₁²/s₂²', dist:'F_{n₁−1, n₂−1}', df:'n₁−1, n₂−1' },
  { q:'Do counts match a claimed distribution?', cond:'all Eᵢ ≥ 5',
    test:'Chi-square goodness of fit', stat:'χ² = Σ(O−E)²/E', dist:'χ²_{k−1−m}', df:'k − 1 − m' },
  { q:'Are two categorical variables related?', cond:'all E_ij ≥ 5',
    test:'Chi-square independence', stat:'χ² = Σ(O−E)²/E', dist:'χ²_{(r−1)(c−1)}', df:'(r−1)(c−1)' },
  { q:'Which of two simple hypotheses?', cond:'simple vs simple',
    test:'Neyman–Pearson LRT', stat:'Λ = L(θ₁)/L(θ₀)', dist:'chosen so the size is α', df:'—' }
];

/* ── A.04 The blank-page checklist — the daily heartbeat deck ──────── */
window.BLANKPAGE = {
  M1:{ title:'Mathematics I', items:[
    { cue:'Point–slope form of a line',                answer:'y-y1=m(x-x1)' },
    { cue:'Slope–intercept form',                      answer:'y=mx+c' },
    { cue:'Intercept form',                            answer:'x/a+y/b=1' },
    { cue:'Normal form',                               answer:'x cos(alpha)+y sin(alpha)=p' },
    { cue:'General form',                              answer:'Ax+By+C=0' },
    { cue:'Two-point form',                            answer:'y-y1=((y2-y1)/(x2-x1))(x-x1)' },
    { cue:'Point-to-line distance',                    answer:'|A x0+B y0+C|/sqrt(A^2+B^2)' },
    { cue:'The quadratic formula',                     answer:'x=(-b±sqrt(b^2-4ac))/(2a)' },
    { cue:'Vertex of a parabola',                      answer:'(-b/(2a),-D/(4a))' },
    { cue:'Sum and product of the roots',              answer:'alpha+beta=-b/a, alpha*beta=c/a' },
    { cue:'The discriminant',                          answer:'b^2-4ac' },
    { cue:'Remainder theorem',                         answer:'p(a)' }
  ]},
  S1:{ title:'Statistics I', items:[
    { cue:'Sample variance s² (definition form)',      answer:'(1/(n-1))sum(x_i-xbar)^2' },
    { cue:'Sample variance s² (computational form)',   answer:'(sum x_i^2-n xbar^2)/(n-1)' },
    { cue:'The 1.5·IQR outlier fences',                answer:'Q1-1.5*IQR and Q3+1.5*IQR' },
    { cue:'Covariance',                                answer:'(1/(n-1))sum(x_i-xbar)(y_i-ybar)' },
    { cue:'Correlation r',                             answer:'cov(x,y)/(s_x s_y)' },
    { cue:'Regression slope b₁',                       answer:'b1=r*s_y/s_x' },
    { cue:'Regression intercept b₀',                   answer:'b0=ybar-b1*xbar' },
    { cue:'ⁿPᵣ',                                       answer:'n!/(n-r)!' },
    { cue:'ⁿCᵣ',                                       answer:'n!/(r!(n-r)!)' },
    { cue:'Arrangements with repeats',                 answer:'n!/(n1! n2! ... nk!)' },
    { cue:'Addition rule for probability',             answer:'P(A)+P(B)-P(A∩B)' },
    { cue:'Conditional probability',                   answer:'P(A∩B)/P(B)' },
    { cue:'Law of total probability',                  answer:'P(A)=sum P(B_i)P(A|B_i)' },
    { cue:'Bayes’ rule',                               answer:'P(B_j)P(A|B_j)/sum_i P(B_i)P(A|B_i)' }
  ]},
  M2:{ title:'Mathematics II', items:[
    { cue:'Rank–nullity theorem',                      answer:'rank(A)+nullity(A)=n' },
    { cue:'Consistency: no solution when…',            answer:'rank(A)<rank([A|b])' },
    { cue:'Consistency: unique solution when…',        answer:'rank(A)=rank([A|b])=n' },
    { cue:'Row space basis from the RREF',             answer:'the non-zero rows of the RREF' },
    { cue:'Column space basis from the RREF',          answer:'the columns of the ORIGINAL A in the pivot positions' },
    { cue:'Cauchy–Schwarz',                            answer:'|<u,v>|<=||u|| ||v||' },
    { cue:'Projection onto a vector u',                answer:'(<v,u>/<u,u>)u' },
    { cue:'Projection matrix onto C(A)',               answer:'P=A(A^T A)^-1 A^T' },
    { cue:'The two properties of a projection matrix', answer:'P^2=P and P^T=P' },
    { cue:'The Gram–Schmidt step',                     answer:'w_j=v_j-sum_{i<j}(<v_j,w_i>/<w_i,w_i>)w_i' },
    { cue:'Orthogonal matrix condition',               answer:'Q^T Q=I, so Q^-1=Q^T' },
    { cue:'The normal equations',                      answer:'A^T A xhat=A^T b' }
  ]},
  S2:{ title:'Statistics II', items:[
    { cue:'Var(X) computational form',                 answer:'Var(X)=E[X^2]-(E[X])^2' },
    { cue:'Var(X ± Y)',                                answer:'Var(X)+Var(Y)±2Cov(X,Y)' },
    { cue:'Markov’s inequality',                       answer:'P(X>=a)<=E[X]/a' },
    { cue:'Chebyshev’s inequality',                    answer:'P(|X-mu|>=k sigma)<=1/k^2' },
    { cue:'E[X̄] and Var(X̄)',                          answer:'E[Xbar]=mu, Var(Xbar)=sigma^2/n' },
    { cue:'The CLT statement',                         answer:'(Xbar-mu)/(sigma/sqrt n) -> N(0,1) in distribution' },
    { cue:'Distribution of (n−1)S²/σ²',                answer:'chi-squared with n-1 df' },
    { cue:'Distribution of (X̄−μ)/(S/√n)',             answer:'t with n-1 df' },
    { cue:'MSE decomposition',                         answer:'MSE=Var(thetahat)+Bias^2' },
    { cue:'CI for μ, σ unknown',                       answer:'xbar ± t_{alpha/2,n-1} s/sqrt(n)' },
    { cue:'CI for a proportion',                       answer:'phat ± z_{alpha/2} sqrt(phat(1-phat)/n)' },
    { cue:'Definition of the p-value',                 answer:'P(statistic at least as extreme as observed | H0 true)' },
    { cue:'Pooled variance s_p²',                      answer:'((n1-1)s1^2+(n2-1)s2^2)/(n1+n2-2)' },
    { cue:'Chi-square count statistic',                answer:'sum (O_i-E_i)^2/E_i' }
  ]}
};

/* The distribution table is the handbook's one compound node: 13 rows × the
   examinable columns. Rather than treat the Vault as a special case, generate
   a real atom per cell so A.01 integrity genuinely tracks table mastery and
   the Vault schedules like everything else. */
(function generateVaultAtoms(){
  const cols = [{ i:1, k:'mean', label:'Mean' }, { i:2, k:'var', label:'Variance' }];
  const out = [];
  VAULT.discrete.concat(VAULT.continuous).forEach(row => {
    const slug = row.name.replace(/[^A-Za-z0-9]/g, '').slice(0, 12);
    row.atomIds = [];
    row.cells.forEach((cell, ci) => {
      const col = cols.find(c => c.i === ci);
      if(!col || cell === '—'){ row.atomIds[ci] = null; return; }
      const id = 'A.01.v' + slug + '.' + col.k;
      row.atomIds[ci] = id;
      out.push({
        id:id, clause:'A.01', course:'AP', yield:3, type:'FORMULA',
        prompt:col.label + ' of ' + row.name, answer:cell, vault:true
      });
    });
  });
  window.CORPUS = (window.CORPUS || []).concat(out);
})();

/* ── Appendix atoms ────────────────────────────────────────────────── */
window.CORPUS = (window.CORPUS || []).concat([

/* A.01 — high-value single facts drawn from the distribution table */
{ id:'A.01.c01', clause:'A.01', course:'AP', yield:3, type:'CLASSIFY',
  prompt:'Which distribution has mean equal to its variance?',
  options:['Poisson(λ)','Binomial(n,p)','Geometric(p)','Exponential(λ)'], answer:0 },
{ id:'A.01.c02', clause:'A.01', course:'AP', yield:3, type:'CLASSIFY',
  prompt:'Mean and variance of χ²_k are…', options:['k and 2k','2k and k','k and k','k/2 and k'], answer:0 },
{ id:'A.01.c03', clause:'A.01', course:'AP', yield:3, type:'CLASSIFY',
  prompt:'Variance of Geometric(p) is…', options:['(1−p)/p²','1/p','p(1−p)','1/p²'], answer:0 },
{ id:'A.01.c04', clause:'A.01', course:'AP', yield:3, type:'CLASSIFY',
  prompt:'The t_k distribution’s MGF…', options:['does not exist','is (1−2t)^(−k/2)','is e^(t²/2)','is k/(k−2)'], answer:0 },
{ id:'A.01.c05', clause:'A.01', course:'AP', yield:3, type:'CLASSIFY',
  prompt:'Which models sampling WITHOUT replacement?',
  options:['Hypergeometric','Binomial','Poisson','Geometric'], answer:0 },
{ id:'A.01.f01', clause:'A.01', course:'AP', yield:3, type:'FORMULA',
  prompt:'MGF of the Normal(μ, σ²)', answer:'e^(mu t+sigma^2 t^2/2)', accept:['exp(μt + σ²t²/2)'] },
{ id:'A.01.f02', clause:'A.01', course:'AP', yield:3, type:'FORMULA',
  prompt:'Mean and variance of Gamma(α, β)', answer:'alpha/beta and alpha/beta^2', accept:['α/β, α/β²'] },
{ id:'A.01.f03', clause:'A.01', course:'AP', yield:3, type:'FORMULA',
  prompt:'Mean of Beta(α, β)', answer:'alpha/(alpha+beta)', accept:['α/(α+β)'] },
{ id:'A.01.f04', clause:'A.01', course:'AP', yield:3, type:'FORMULA',
  prompt:'χ²_k written as a Gamma distribution', answer:'Gamma(k/2,1/2)', accept:['Gamma with α=k/2, β=1/2'] },
{ id:'A.01.f05', clause:'A.01', course:'AP', yield:3, type:'FORMULA',
  prompt:'Variance of t_k, and when it exists', answer:'k/(k-2) for k>2', accept:['k/(k−2), k > 2'] },

/* A.02 — the map itself */
{ id:'A.02.c01', clause:'A.02', course:'AP', yield:3, type:'CLASSIFY',
  prompt:'Mean, σ unknown, normal population → which test?',
  options:['one-sample t, df n−1','one-sample z','paired t','chi-square'], answer:0 },
{ id:'A.02.c02', clause:'A.02', course:'AP', yield:3, type:'CLASSIFY',
  prompt:'Two independent groups, unequal variances → which test?',
  options:['Welch t','pooled t','paired t','F test'], answer:0 },
{ id:'A.02.c03', clause:'A.02', course:'AP', yield:3, type:'CLASSIFY',
  prompt:'Same units measured twice → which test?',
  options:['paired t, df n−1','pooled t','two-sample z','chi-square'], answer:0 },
{ id:'A.02.c04', clause:'A.02', course:'AP', yield:3, type:'CLASSIFY',
  prompt:'Are two variances equal? → which test and df?',
  options:['F, df (n₁−1, n₂−1)','chi-square, df n−1','t, df n−2','z'], answer:0 },
{ id:'A.02.c05', clause:'A.02', course:'AP', yield:3, type:'CLASSIFY',
  prompt:'Are two categorical variables related? → which df?',
  options:['(r−1)(c−1)','k−1−m','n−1','r+c−2'], answer:0 },

/* A.03 — notation */
{ id:'A.03.dsc01', clause:'A.03', course:'AP', yield:1, type:'DISCRIM',
  left:'Population (parameter)', right:'Sample (statistic)',
  items:[{t:'μ', side:'L'},{t:'x̄', side:'R'},{t:'σ²', side:'L'},{t:'s²', side:'R'},
         {t:'p', side:'L'},{t:'p̂', side:'R'},{t:'ρ', side:'L'},{t:'r', side:'R'},
         {t:'N', side:'L'},{t:'n', side:'R'}] },
{ id:'A.03.c01', clause:'A.03', course:'AP', yield:1, type:'CLASSIFY',
  prompt:'N(A) and C(A) denote…',
  options:['null space and column space','norm and cardinality','nullity and rank','neither'], answer:0 },
{ id:'A.03.c02', clause:'A.03', course:'AP', yield:1, type:'CLASSIFY',
  prompt:'The symbol ⟶d means…',
  options:['converges in distribution','converges in probability','is distributed as','differentiates'], answer:0 },
{ id:'A.03.c03', clause:'A.03', course:'AP', yield:1, type:'CLASSIFY',
  prompt:'⊥ is read as…',
  options:['orthogonal to / independent of','perpendicular distance','the null set','parallel to'], answer:0 },
{ id:'A.03.c04', clause:'A.03', course:'AP', yield:1, type:'CLASSIFY',
  prompt:'z_α denotes…',
  options:['the upper α critical value','the standard normal CDF','a sample mean','the α-th percentile from the left'], answer:0 },
{ id:'A.03.c05', clause:'A.03', course:'AP', yield:1, type:'CLASSIFY',
  prompt:'ker T and im T are…',
  options:['the kernel and image of a transformation','kernel and identity','both subspaces of V','undefined'], answer:0 }

]);
