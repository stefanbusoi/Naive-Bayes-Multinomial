# Documentație Proiect --- Clasificare Text cu Naive Bayes

## 1. Introducere

Acest proiect implementează un clasificator Naive Bayes Multinomial
pentru descrieri de jocuri video. Scopul este de a determina genurile
unui joc pe baza textului din descrierea sa.

Proiectul folosește un set de date prelucrat manual, conținând descrieri
text și genurile asociate fiecărui joc.

------------------------------------------------------------------------

## 2. Modelul Folosit: Naive Bayes Multinomial

Modelul Naive Bayes clasifică un text nou pe baza următoarei formule:

\[ P(C_k \| X) = P(C_k) `\prod`{=tex}\_{i=1}\^n P(x_i \| C_k) \]

unde: - (C_k) este o categorie (ex: Action, RPG), - (x_i) sunt cuvintele
din text, - (P(C_k)) este probabilitatea a priori a categoriei, - (P(x_i
\| C_k)) este probabilitatea apariției cuvântului în categoria
respectivă.

Pentru stabilitate numerică, se folosesc log-probabilități.

------------------------------------------------------------------------

## 3. Structura Codului

### **3.1 Preprocesarea datelor**

-   încărcare CSV
-   transformare genuri în listă
-   tokenizare descriere
-   eliminare stopwords minimale
-   contorizare cuvinte per gen

### **3.2 Calcularea probabilităților**

-   Probabilități a priori ale fiecărui gen
-   Probabilități condiționate cu smoothing (0.01)

### **3.3 Funcția de clasificare**

`testWord(text)`: - tokenizează textul - acumulează log-probabilitățile
pentru fiecare gen - decide dacă un gen este prezent sau nu

------------------------------------------------------------------------

## 4. Împărțirea Datelor

Setul este împărțit astfel: - **Training:** primele 50.000 intrări -
**Test:** restul intrărilor

Această separare asigură evaluarea corectă a modelului.

------------------------------------------------------------------------

## 5. Evaluarea Modelului

Evaluarea se face prin: - numărarea True Positive, True Negative, False
Positive, False Negative - corectitudinea totală (accuracy) -
corectitudine per gen

Formula pentru acuratețe:

\[ Accuracy = `\frac{TP + TN}{TP + TN + FP + FN}`{=tex} \]

------------------------------------------------------------------------

## 6. Instrucțiuni de Utilizare

1.  Rulați scriptul Python.
2.  Alegeți dacă doriți să introduceți manual un text sau să rulați
    direct testarea.
3.  Programul afișează:
    -   genurile prezise,
    -   acuratețea pe setul de test,
    -   metricele pe fiecare gen.

------------------------------------------------------------------------

## 7. Referințe

-   Wikipedia: *Naive Bayes Classifier*
-   Manning, Raghavan, Schütze -- *Introduction to Information
    Retrieval*
-   Documentația Python Pandas și NumPy
