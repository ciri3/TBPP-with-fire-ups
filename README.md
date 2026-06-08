# Appunti per le slide

* gamma = 1 per considerare caso più difficile con due obiettivi con stesso peso (articolo originale si aveva caso semplice con gamma=1/n - conn n = # job - in cui il TBPP con FU si risolve come il TBPP normale: gamma <= 1/n)  
* hanno introdotto solo il mdoello 3 (riduce drasticamente # variabili e vantaggio computazionale significativo)  
* modello 2 riduce var da <=5n^2+n a <=2n^2+n e vincoli da <=4n^2+n a =3n^2+n (reminder: T->TS, 2n->n)  
    * non considerato in computaz. perché peggior LP-bounds, necessari improvements

## Miglioramenti modelli 1 e 2

Obiettivo: armonizzare LP-bounds per avere miglioramenti computazionali dal 2° modello  
Necessaria: nuova formulazione ILP