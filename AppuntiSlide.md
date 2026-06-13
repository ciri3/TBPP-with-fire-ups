# Appunti per le slide

## Struttura presentazione

1. *Into*: problema (problema relativamente nuovo -> stato letteratura attuale)
1. *1° modello*: spiegazione M1
    1. *Ottimizzazioni*: verso M1opt
    1. *Testbed*: A e B
    1. *Confronto*: M1 vs M1opt
1. *2° modello*: spiegazione M2 + M2otp --> confronto
1. *3° modello*: spiegazione M3 + M3otp --> confronto
1. *Confronto*: modelli ottimizzati
1. *Risultati e considerazioni finali*

---

* gamma = 1 per considerare caso più difficile con due obiettivi con stesso peso (articolo originale si aveva caso semplice con gamma=1/n - conn n = # job - in cui il TBPP con FU si risolve come il TBPP normale: gamma <= 1/n)  
* hanno introdotto solo il mdoello 3 (riduce drasticamente # variabili e vantaggio computazionale significativo)  
* modello 2 riduce var da <=5n^2+n a <=2n^2+n e vincoli da <=4n^2+n a =3n^2+n (reminder: T->TS, 2n->n)  
    * non considerato in computaz. perché peggior LP-bounds, necessari improvements

## Miglioramenti modelli 1 e 2

Obiettivo: armonizzare LP-bounds per avere miglioramenti computazionali dal 2° modello  
Necessaria: nuova formulazione ILP

### miglioramenti modello 1

usiamo lit a, b e c

usaimo solo R2 (a) perchè 
“In our internal precalculations, we noticed that only Conditions (18) lead to any benefits in terms of the LP value.”
“Hence, we will only use this set of inequalities in the final numerical experiments, while the others, i.e., categories (19)–(21), are intended to remain in the list for the sake of completeness.”

non usiamo R3 perché:
"According to Table 7, for the instances considered here, the most significant contribution (in terms of the LP value) is given by (R2). More precisely, adding valid inequalities leads to the largest increase of the lower bound, and any configuration containing (R2) has the same LP value."
"So, for the instances considered in this experiment, (R2) is somewhat 'dominating' (R3), which is consistent with the observation made earlier that, in the end, the LP value always matches 2h."

però:
"In any case, we would like to state that the additional (time) effort of the lifting procedure is so low that we still recommend using this reduction, even if it may not be successful."

## miglioramenti modello 2
...

## Osservazioni  
1. M1 è quello con prestazioni più costanti, il più affidabile, benché non sempre il più veloce
    * è il più veloce per istanze molto grandi
2. M2 in generale non è il migliore ma compete e se la cava certe volte (M1 < M2 < M3)
3. M3 è di gran lunga il più veloce per istanze piccole e in generale è una buona euristica per ottenere una soluzione velocemente ma è il più instabile, il meno affidabile dal punto di vista delle prestazioni

## analisi parametri scalabilityTest

Due to limited computational resources, we used a reduced but structured benchmark. For Testbed A, we preserved the factorial structure of the original benchmark while reducing the number of jobs to n∈{25,50,75,100}. For Testbed B, we selected representative values of ∣T∣ and a subset of classes covering different levels of instance difficulty.


# analisi confronti 
Per l'istanza con parametri n = 45, s = 1.0, duration = long, size = low si osservano particolari difficoltà computazionali. In particolare, per alcuni seed (ad esempio 43 e 44) sia le formulazioni base sia quelle ottimizzate dei modelli M1 e M2 raggiungono il time limit senza provare l'ottimalità. Per altri seed della stessa categoria (ad esempio 45), le versioni ottimizzate riescono invece a trovare soluzioni migliori rispetto alle formulazioni base, evidenziando l'efficacia delle tecniche di riduzione adottate pur in presenza di istanze particolarmente difficili.