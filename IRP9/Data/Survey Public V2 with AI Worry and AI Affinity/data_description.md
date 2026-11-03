# Data Description: Public Perception of Explainable AI in Financial Services

## Overview
This document provides a detailed data dictionary and structural overview for the dataset contained in **numeric_dataset.csv**. The metadata, survey logic, and theoretical justifications outlined in this file are derived from the companion technical document, **Survey_Supplement_Public.pdf**.

The dataset captures user experiences, awareness, and preferences regarding automated decision-making and Explainable AI (XAI) within financial applications. It includes branching logic based on user experience and collects demographic data to allow for socio-cultural and demographic subgroup analyses.

---

## Dataset Structure & Branching Logic

The survey utilizes conditional branching based on the respondent's prior experience with AI financial apps (specifically Question 18). 

*   **Path 1 (Received an Explanation):** Respondents answer Q19–Q24, then proceed to Q28–Q40.
*   **Path 2 (Did Not Receive an Explanation):** Respondents answer Q25–Q27, then proceed to Q28–Q40.
*   **All Respondents:** Answer Q1–Q17, Q28–Q40, and Q41–Q49.

*Note: Variables 12 and 37 act as strict attention checks for data quality control.*

---

## Data Dictionary

### Section 1: Consent & AI Awareness
| Var ID | Topic / Question | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **1** | Consent | Binary Categorical | Ethical prerequisite confirming voluntary participation. |
| **2-6** | AI Awareness | Ordinal Categorical | Establishes baseline awareness of AI use in finance, opacity, and XAI problems. |
| **7** | Tradeoff Preferences | Discrete Numerical (0-10) | Elicits preference for interpretability vs. predictive performance. |
| **8-11** | General AI Attitude | Ordinal Likert | Captures general attitudes toward artificial intelligence (AIAS-4 adapted). |
| **12** | Attention Check 1 | Categorical | Data quality control item. |

### Section 2: AI Finance Experience
| Var ID | Topic / Question | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **13** | AI App Usage | Categorical | Screening item: Identifies direct experience with AI-enabled financial apps. |
| **14** | Platform Name | Open-ended text | The name or description of the specific financial platform used. |
| **15** | Interaction Role | Multiple-response Categorical | Distinguishes user roles (e.g., app user, business user, developer). |
| **16** | Automated Activities | Multiple-response Categorical | Identifies specific financial activities involving automated decision-making. |
| **17** | Other Activities | Open-ended text | Text field for activities not listed in Q16. |
| **18** | Received Explanation? | Categorical Branching | Determines if the app provided an explanation for a financial decision. (Dictates routing to Q19 or Q25). |

### Section 3A: Explanation Quality & Trust (Condition: Received Explanation)
*Only answered if Q18 = Yes.*

| Var ID | Topic / Question | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **19** | Adaptive Knowledge Match | Ordinal Likert | Measures if the explanation matched the user's expertise level. |
| **20** | Comprehensibility | Ordinal Likert | Measures if the user understood how and why the recommendation was made. |
| **21** | Sufficiency of Detail | Ordinal Likert | Evaluates explanation usability (not too much/little detail). |
| **22** | Trust Calibration | Ordinal Likert | Measures if the explanation helped the user appropriately judge/trust the decision. |
| **23** | Desired Improvements | Open-ended text | Identifies shortcomings and desired improvements to the explanation. |
| **24** | Trust Impact (Hypothetical) | Discrete Numerical | How much trust would have changed if improvements were implemented. |

### Section 3B: Experience Without Explanation (Condition: No Explanation)
*Only answered if Q18 = No.*

| Var ID | Topic / Question | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **25** | Emotional/Trust Response | Ordinal Categorical | Measures feelings toward an outcome provided without explanation. |
| **26** | Desired Explanation Type | Open-ended text | Identifies the explanation content/format the respondent hoped to see. |
| **27** | Trust Impact (Hypothetical) | Discrete Numerical | Expected effect on trust if a preferred explanation had been provided. |

### Section 4: Explanation Preferences (All Respondents)
| Var ID | Topic / Question | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **28** | Required Explanation Cases | Multiple-response Categorical | Identifies scenarios where respondents believe AI explanations should be mandatory. |
| **29** | Decision Process (Feature) | Ordinal Importance | Importance of knowing how inputs (e.g., spending) affected the result. |
| **30** | Main Influencing Factors | Ordinal Importance | Importance of knowing primary factors (e.g., income, payment history). |
| **31** | Expert/Benchmark Comparison | Ordinal Importance | Importance of comparing results to market averages or official criteria. |
| **32** | Confidence / Reliability | Ordinal Importance | Importance of seeing the AI's confidence level in its prediction. |
| **33** | Human vs. Automated Process | Ordinal Importance | Importance of understanding which parts of the process involved humans. |
| **34** | Error Detection & Correction | Ordinal Importance | Importance of knowing how the app handles and corrects incorrect data. |
| **35** | Decision Maker (Human/AI) | Ordinal Importance | Importance of knowing if a human or AI made the final decision. |
| **36** | Risks and Consequences | Ordinal Importance | Importance of being informed about risks if the AI makes an error. |
| **37** | Attention Check 2 | Categorical | Data quality control ("Please select 'Don't Know'"). |
| **38** | Recourse / Advice | Ordinal Importance | Importance of advice on how to improve the outcome (algorithmic recourse). |
| **39** | Personalized Explanation | Ordinal Importance | Importance of explanations tailored to the specific user case. |
| **40** | Data Privacy & Usage | Ordinal Importance | Importance of knowing how user data is protected and used by the AI. |

### Section 5: Responsibility, Governance, and Demographics
| Var ID | Topic / Question | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **41** | Accountability | Single-response Categorical | Identifies who should ensure financial AI systems are transparent. |
| **42** | Other Governing Body | Open-ended text | Allows specification of accountability actors not listed in Q41. |
| **43** | Gender | Categorical | Demographic variable for subgroup analysis. |
| **44** | Age | Categorical | Demographic variable for age group comparisons. |
| **45** | Country | Categorical / Text | Identifies national/regulatory context (useful for cross-cultural analysis). |
| **46** | Job Level | Categorical | Measures professional seniority. |
| **47** | Job Title (Other) | Open-ended text | Custom field for unlisted job titles. |
| **48** | Field of Study | Categorical | Assesses academic background (students only). |
| **49** | Occupational Sector | Categorical | Identifies industry sector to gauge exposure to financial technologies. |

---
*Reference Material: Please consult **Survey_Supplement_Public.pdf** for detailed theoretical frameworks, literature citations, and academic justifications associated with these constructs.*