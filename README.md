# Installation Process

Welcome to our repository! To begin contributing to the Individual Research Projects, each team member must first create a GitHub account. 
Once your account is set up, please send an email to [gabin.taibi@bfh.ch](mailto:gabin.taibi@bfh.ch) to be added to the team and the appropriate repository. 
After being added, you can handle your code through Git using the command line, any Git GUI (like SmartGit, SourceTree, etc.), or GitHub Desktop, which is more suitable for beginners to Git. The next sections will guide you through setting up SSH keys.

## GitHub Desktop

GitHub Desktop is an application that simplifies the interaction with GitHub repositories. It provides a graphical interface to manage your repositories without using command-line tools.

### Install GitHub Desktop:
1. **Download GitHub Desktop**: Visit [GitHub Desktop](https://desktop.github.com/) and download the application for your operating system.
2. **Install the Application**: Run the downloaded installer and follow the on-screen instructions.
3. **Log in to Your GitHub Account**: Open GitHub Desktop and sign in with your GitHub credentials.
4. **Clone the Repository**: After logging in, you can clone the repository to your local machine. Go to `File` > `Clone Repository`, and select the repository you want to work on.

## SSH Keys

SSH keys are a way to identify trusted computers, without involving passwords. Setting up SSH keys will allow you to push and pull changes securely.

### Steps to Set Up SSH Keys:
1. **Check for Existing SSH Keys**: First, check if you already have an SSH key. Open a terminal and run `ls -al ~/.ssh`. Look for files named `id_rsa.pub` or `id_ed25519.pub`.
2. **Generate a New SSH Key**: If you don't have an SSH key, generate one using `ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`. Replace "your_email@example.com" with your GitHub email address.
3. **Add Your SSH Key to the SSH-Agent**: Run `eval "$(ssh-agent -s)"` and then `ssh-add ~/.ssh/id_rsa`.
4. **Add SSH Key to Your GitHub Account**: Copy your SSH key to the clipboard with `pbcopy < ~/.ssh/id_rsa.pub` (Linux/MacOS) or `clip < ~/.ssh/id_rsa.pub` (Windows). In GitHub, go to `Settings` > `SSH and GPG keys` > `New SSH key`, and paste your key.



# The Repository

This repository is structured into several key folders, each intended for different types of content.

## Paper Folder

- **Purpose**: This folder is for storing and sharing academic papers related to our project.
- **Usage**: To add a paper, simply place your document in this folder and commit your changes. Supported formats include `.pdf`, `.docx`, etc.

## Presentations Folder

- **Purpose**: Contains presentations, such as slideshows or posters, relevant to our work.
- **Usage**: Add your presentations in formats like `.pptx`, `.key`, or `.pdf`. Make sure to name them clearly for easy identification.

## Code Folder

- **Purpose**: This is where all code-related materials are stored, you can add pure code and data (in the data folder)
- **Usage**: You can add source code, scripts, and related documentation here. We primarily use `.py` (Python) files, but other formats are accepted as needed for the project.



# Additional Tips

- **Committing Changes**: When you make changes, ensure you 'commit' them with a clear message describing what you've done. This helps others understand the history of changes.
- **Pulling Latest Changes**: Regularly 'pull' from the repository to ensure you have the latest version of all files.
- **Branching for Safety**: Since you will have access to the entire WP repository, which contains folders for other IRPs, it's important to be cautious. Avoid modifying other people's work by creating separate branches for your contributions. This practice helps isolate your changes and prevents accidental alterations to others' work.
- **Asking for Help**: If you're stuck or need help, don’t hesitate to ask. Collaboration is key to our success!



# WP3_eXplainable_and_Fair_AI

- Lead Beneficiary: Bern Business School.
- Researchers involved: See link
- Bern Business School (Lead): Adam Kurpisz, Branka Hadji Misheva, Christian Hopp
- University of Naples Federico II: Francesco Palumbo, Alfonso Iodice D'Enza, Maria Iannario, Antonio Pescapè
- Active period: From M4 to M48.
- Activity type: Research.
- Early Stage Researchers involved: 9, 16 & 17.


## Objectives
The WP will work towards a unifying framework of explainability for AI models applied to financial use cases.
	3.1. To answer the main research questions on solving explainability deployment hurdles for financial applications.
	3.2. To demonstrate the proposed framework for audience-dependent explanations, through use cases (SWE, INT, ROY).
	3.3. To disseminate the knowledge, validated by an international research centre (FRA, ECB, ARC)


## Description
WP 3 is led by BFH and supported by all partners. The work is divided into the following tasks:

- Task 3.1. Technical coordination. Monitoring the related IRPs, store the output generated in a location accessible to the entire network.
- Task 3.2. Support the research training for all assigned ESRs and contribute to advanced training content
- Task 3.3. To provide global and local post hoc explainability techniques that address the explainability needs of different stakeholders.
- Task 3.4. To propose explainability functions, tailored for financial time series, preserving the non-stationary dependence structure.
- Task 3.5. Develop new portfolio optimization models that address challenges of incorporating fairness considerations into investments.
- Task 3.6. Disseminate, communicate and exploit the results (Conferences, OS Day, policy paper, two prototypes, use case, media coverage)

## Deliverables
- D.3.1 Documentation of explainable AI methods: Documentation of test setups for applying explainable AI methods (Due in M48)
- D.3.2 Technical report on trustworthy AI methods: Technical report showing the achievements on trustworthy and fair AI models (Due in M48)
D.3.3 Summary report on time-series explainability: Summary report on all results and impacts related to explainability for time-series (Due in M24)