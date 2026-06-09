# Appunti per le slide

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