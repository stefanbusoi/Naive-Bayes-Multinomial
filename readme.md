# Documentație Proiect --- Clasificare Descriere Joc in functie de tipul de joc

## 1. Introducere

Acest proiect implementează  Naive-Bayes-Multidimentional si Log-likelihood ratio naive biass
pentru descrieri de jocuri video. Scopul este de a determina genurile
unui joc pe baza textului din descrierea sa.

Proiectul folosește un set de date care contine descrieri
text și genurile asociate fiecărui joc.


## 2. Modelul Matematic folosit 
### a)Likelihood-ratio naive biass

Modelul clasifică un text nou pe baza următoarei formule:
```math
 P(C_k | Tokens) \propto P(C_k) \prod_{t \in Words} \frac{P(t | C_k)}{P(t)} 
```

Pentru stabilitate numerică, se face calculul logaritmat

---
### b) Naive Biass multinomial 
Modelul clasifică un text nou pe baza următoarei formule:

```math
 P(C_k | Tokens) \propto P(C_k) \prod_{t \in Words} {P(t | C_k)} 
```
pentru stabilitate numerica am facut calculul logaritmat
am normalizat ```P(C_k|tokens)``` impartind la ```len(Tokens)```

---
Pentru ambele formule ```P(C_k)``` este o categorie (ex: Action, RPG), ```Tokens``` e multimea cuvintelor
din text, ```P(C_k)``` este probabilitatea a priori a categoriei, ```(P(t|C_k))``` este probabilitatea apariției cuvântului în categoria respectivă.

------------------------------------------------------------------------

## 3. Structura Codului

### **3.1 Preprocesarea datelor**

-   încărcare CSV
-   transformare genuri în listă
-   tokenizare descriere
-   contorizare cuvinte per gen

### **3.2 Calcularea probabilităților**

-   Probabilități a priori ale fiecărui gen
-   Probabilități condiționate cu smoothing (0.01)

### **3.3 Funcția de clasificare**

`testLLRNaiveBiass` sau ` testNaiveBiassMultinominal(str)`: tokenizează textul , acumulează log-probabilitățile
pentru fiecare gen si decide dacă un gen este prezent sau nu

------------------------------------------------------------------------

## 4. Împărțirea Datelor

Setul este împărțit astfel: 
- **Training:** primele 50.000 intrări 
- **Test:** restul intrărilor

------------------------------------------------------------------------

## 5. Evaluarea Modelului

Evaluarea se face prin numărarea:
- True Positive 
- True Negative
- False Positive 
- False Negative 
- corectitudine
- corectitudinea totală (accuracy) 

Formula pentru acuratețe:
```math
Accuracy = \frac{TP + TN}{TP + TN + FP + FN} 
```
------------------------------------------------------------------------

## 6. Instrucțiuni de Utilizare

1.  Rulați scriptul ```program.py```
2.  Alegeți dacă doriți să introduceți manual un text sau să rulați testarea.
3.  Programul afișează:
    -   genurile prezise,
    -   acuratețea pe setul de test,
    -   metricele pe fiecare gen.

------------------------------------------------------------------------

## 7. Referințe

-   Wikipedia: *Naive Bayes Classifier*
-   Documentația Python Pandas și NumPy
