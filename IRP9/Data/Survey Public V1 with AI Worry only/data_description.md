# Data Description: Global Translated Survey on XAI in Finance

## Overview

This document provides a comprehensive data dictionary and structural overview for the dataset contained in **survey_data_global_translated.csv**.

**IMPORTANT NOTE ON TRANSLATION & GEOGRAPHY:**
The original survey instrument, referenced in the file **Survey_PublicV1_IT.pdf**, was primarily designed and administered in Italian because the primary target demographic was based in Italy. While the survey was technically available in other languages, there were very few respondents from other individual countries. Because of this low sample size outside of Italy, these non-Italian locations were grouped together into an "Other" category. Consequently, all data, column headers, categorical variables, and open-ended text responses across the entire dataset have been fully translated into English to facilitate a standardized, unified analysis.

The survey explores user familiarity, preferences, and trust regarding Artificial Intelligence (AI) and Explainable AI (XAI) within financial applications.

## Dataset Structure & Branching Logic

The survey employs conditional branching based on the respondent's prior experience receiving an AI explanation:

* **Path A (Received an Explanation):** If respondents indicate they received an explanation from a financial app, they are routed to evaluate the quality of that explanation and how it impacted their trust.

* **Path B (Did Not Receive an Explanation):** If respondents did not receive an explanation, they are asked about their emotional/trust response to the "black box" decision and what type of explanation they would have preferred.

* **Common Paths:** All respondents answer the introductory familiarity questions, the core preference questions on XAI features, and demographic information.

## Data Dictionary

### Section 1: Consent & AI Familiarity

*(Translated from "Tu e l'Intelligenza Artificiale")*

| Variable / Topic | Data Type | Description | 
| ----- | ----- | ----- | 
| **Consent** | Binary | Confirmation of age (>18) and voluntary participation. | 
| **Fam_Banks_AI** | Ordinal (1-5) | Familiarity with banks using AI for credit/financial decisions. (1=Not at all, 5=Very familiar) | 
| **Fam_BlackBox** | Ordinal (1-5) | Familiarity with "black box" models in finance. | 
| **Fam_RightToExplain** | Ordinal (1-5) | Awareness of the right to request explanations for AI financial decisions. | 
| **Fam_Acc_vs_Exp** | Ordinal (1-5) | Awareness that highly accurate AI often struggles to explain its decisions. | 
| **Fam_Interpretability** | Ordinal (1-5) | Awareness of the difference between inherently interpretable AI and post-hoc explanations. | 
| **Pref_Tradeoff** | Numeric (0-10) | Preference scale: `0` (Maximum accuracy, no explanation) to `10` (Clear explanations, slightly less accurate). | 

### Section 2: App Experience

*(Translated from "La tua esperienza con l'app finanziarie")*

| Variable / Topic | Data Type | Description | 
| ----- | ----- | ----- | 
| **Used_AI_App** | Categorical | Has the user utilized an AI-powered financial app? (Yes / No / Not sure) | 
| **App_Name_URL** | Open-Text | Name, description, or URL of the platform used. | 
| **User_Role** | Multi-Categorical | Respondent's role: App User, Business User, Regulator/Supervisor, Creator/Developer. | 
| **AI_Activities** | Multi-Categorical | Tasks automated by the app (e.g., Loan approval, Fraud detection, Investments, ESG scoring, Robo-advisor). | 
| **Explanation_Provided** | Categorical | **\[Branching Point\]** Did the app provide an explanation for its decision? (Yes / No / Don't know) | 

### Section 3A: Evaluating the Explanation Received

*(Conditional: Only if `Explanation_Provided` = "Yes")*

| Variable / Topic | Data Type | Description | 
| ----- | ----- | ----- | 
| **Exp_Knowledge_Match** | Ordinal (1-5) | The explanation was adapted to my knowledge/experience level. | 
| **Exp_Comprehension** | Ordinal (1-5) | I understood how and why the recommendation was made. | 
| **Exp_Detail_Level** | Ordinal (1-5) | The explanation had the right amount of detail (not too much/little). | 
| **Exp_Trust_Eval** | Ordinal (1-5) | The explanation helped me evaluate whether to trust the decision. | 
| **Desired_Improvements** | Open-Text | How the explanation could have been more useful or better adapted. | 
| **Trust_Change_If_Imp** | Numeric (0-10) | How much trust would increase if those improvements were made (`0`=No change, `10`=Much more trust). | 

### Section 3B: Experience Without Explanation

*(Conditional: Only if `Explanation_Provided` = "No")*

| Variable / Topic | Data Type | Description | 
| ----- | ----- | ----- | 
| **Trust_No_Exp** | Categorical | Feelings toward the decision without explanation (Trusted it, Trusted partly, Uncertain, Did not trust at all, Don't remember). | 
| **Desired_Exp_Type** | Open-Text | What type of explanation the respondent hoped to receive. | 
| **Trust_Change_If_Prov** | Numeric (0-10) | How much trust would have increased if the desired explanation was provided (`0` to `10`). | 

### Section 4: What Makes a Good Explanation

*(Translated from "Cosa comprende una buona spiegazione" - Answered by All)*

| Variable / Topic | Data Type | Description | 
| ----- | ----- | ----- | 
| **Mandatory_Exp_Cases** | Multi-Categorical | Scenarios where AI *must* be required to explain itself (e.g., Loans, AML, Insurance, ESG, All). | 
| **Imp_Clear_Logic** | Ordinal (1-5) | Importance of a clear explanation of how the result was determined. | 
| **Imp_Main_Factors** | Ordinal (1-5) | Importance of knowing the main influencing factors (e.g., income, habits). | 
| **Imp_Expert_Bench** | Ordinal (1-5) | Importance of comparing results with expert opinions/standard benchmarks. | 
| **Imp_Reliability** | Ordinal (1-5) | Importance of seeing the AI's confidence/reliability score. | 
| **Imp_Human_vs_Auto** | Ordinal (1-5) | Importance of understanding which parts of the process were automated vs. human. | 
| **Imp_Error_Handling** | Ordinal (1-5) | Importance of understanding how errors are detected and corrected. | 
| **Imp_Decision_Maker** | Ordinal (1-5) | Importance of knowing if the final reviewer was human or AI. | 
| **Imp_Risks_Conseq** | Ordinal (1-5) | Importance of being informed of risks if the AI makes an error. | 
| **Imp_Recourse** | Ordinal (1-5) | Importance of receiving advice on how to improve the outcome (algorithmic recourse). | 
| **Imp_Personalization** | Ordinal (1-5) | Importance of receiving an explanation personalized to their specific case. | 
| **Imp_Data_Privacy** | Ordinal (1-5) | Importance of knowing how personal data is used and protected. | 
| **Responsibility** | Categorical | Who should guarantee transparency? (Financial Institutions, Regulators, Civil Society, All, Other). | 
| **Responsibility_Other** | Open-Text | Text field if "Other" is selected for transparency responsibility. | 

### Section 5: Demographics

*(Translated from "Demografia")*

| Variable / Topic | Data Type | Description | 
| ----- | ----- | ----- | 
| **Gender** | Categorical | Male, Female, Non-binary, Prefer not to say. | 
| **Age_Group** | Categorical | 18-24, 25-34, 35-44, 45-54, 55-64, 65+. | 
| **Country** | Categorical | Country of residence or primary work (Grouped as Italy / Other). | 
| **Province_IT** | Categorical | Italian province (Only applicable if Country = Italy). | 
| **Education_Level** | Categorical | Highest level of education attained (e.g., High School, Bachelor's, Master's, PhD). | 
| **Professional_Level** | Categorical | Job seniority (e.g., Student, Junior, Employee, Manager, Executive). | 
| **Occupational_Sector** | Categorical | Primary industry of employment (e.g., Finance, Tech, Healthcare, Public Admin). | 
| **Field_of_Study** | Categorical | Academic background/field of study. | 
