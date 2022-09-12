##DL PER SERIE TEMPORALI
L'idea e' di testare diverse architetture per comprendere quali meccanismi sono piu' promettenti in situazioni reali. Un problema tipico comprende:
1- una o piu' serie temporali di cui fare la predizione nel futuro anche a piu' step --> multioutput multistep
2- l'input del modello contiene diverse cose, la piu' importante di solito e' lo storico della variabile y. Poi ci sono alcuni segnali del passato 'paralleli' alla y--> covariate. Generalmente non sono note nel futuro--> cosa possiamo farcene? Mi possono indicare uno stato di partenza, ma come lo uso?
3- ci sono delle variabili note nel futuro. ES: programmazione di produzione, forecast di variabili metereologiche
4- ci sono le variabili categoriche, generalmente sia per il passato che per il futuro--> come usarle? ES: il giorno della settimana lo posso trasformare con sin e cos per avere la ciclicita' e un dato continuo, ma le feste? Utilizzare embedding aiuta? Ci sono altri tipi di approcci per le variabili categoriche?

In questo setup generico ci sono diverse architetture da provare:
1- LSTM/GRU plain--> pro: facili da implementare, contro non so come usare le info future
2- LSTM encoder+decoder pro: facili da implementare anche se ci sono delle scelte da fare, posso utilizzare le info del futuro nel decoder, contro: difficile da allenare?
3- NeuralProphet: per ora l'unico modello out-of-the-box che mi ha dato soddisfazione (https://github.com/ourownstory/neural_prophet)
4- Transformer su dati float? Da esplorare
5- Tokenization + tranformer su dati categorici? Che io sappia nessuno ci ha lavorato


Problematiche riscontrate finora:
1 -Non so bene dove sia meglio mettere gli embedding in 1,2 e 4
2- Bisogna fare attenzione a cosa abbiamo in inference
3- Aggiungere layer conv1d dovrebbe aiutare perche' non so bene quale e' il lag delle varie dipendenze. Il transformer dovrebbe automaticamente gestire questa cosa
4- Come posso avere degli intervalli di confidenza su y? Esistono layers che implementano in concetto di distribuzione dei pesi --> pytorch-blitz (https://github.com/piEsposito/blitz-bayesian-deep-learning MA E' DI UN ANNO FA), ce ne sono altri? Possiamo usare il modello 5 e le probabilita' sui token per fare un ensamble e/o dare un intervallo di confidenza piu' veritiero
5- Generalmente le TS possono avere un tot di bias (es: trend) da tenere sotto controllo

SCHEMA DI APPROCCIO
possiamo decidere insieme a queli di queste domande rispondere e come. Possiamo anche cominciare da un esempio di dati reali per capire di cosa si parla. Io credo che implementero' un generatore di dati per testare le architetture in modo da avere delle serie sotto controllo. Si puo partire da qui https://github.com/Nike-Inc/timeseries-generator
In ogni caso serve implementare e testare almeno 1 e 2 in modo da avere tutto sotto controllo soprattutto per quanto riguarda le variabili categoriche. A tendere vorrei arrivare a 5
