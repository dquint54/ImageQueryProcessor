
# ImageQueryProcessor

## Description

is a tool designed to bridge the gap between textual queries and visual content, developed with an integration into Databricks for enhanced data processing and analysis capabilities. By leveraging advanced embedding techniques, OpenAI's powerful models, and Databricks' unified analytics platform, it retrieves and displays relevant images from an S3 bucket based on text descriptions. This project aims to simplify the process of finding visual content that matches textual descriptions, enhancing user experience in various applications such as digital asset management, content discovery, and more, while fully utilizing Databricks for scalable data processing and machine learning workflows.

## Installation

To set up the project, follow these steps:

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/AIEnhancedImageRetrieval.git
cd AIEnhancedImageRetrieval
```

2. **Install Dependencies**
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

## Database Setup in ChromaDB

To store and retrieve image embeddings and document data efficiently, setting up a database in ChromaDB is required.

1. **Create a ChromaDB Account**
   - Visit [ChromaDB's website](https://www.chromadb.com) and sign up for an account.

2. **Create a New Database**
   - Once logged in, navigate to the dashboard and create a new database. Note down the database credentials and endpoint.

3. **Configure Database Access**
   - Use the provided credentials to configure database access within the application. It's recommended to use environment variables for storing these credentials securely.

```bash
export CHROMADB_ENDPOINT='your_chromadb_endpoint'
export CHROMADB_API_KEY='your_chromadb_api_key'
```

4. **Initialize Your Database**
   - Utilize the initialization script provided in the repository to set up the necessary tables and schemas in your ChromaDB database.

3. **Configure AWS Credentials**
Ensure you have AWS CLI installed and configured with your AWS account. See [AWS CLI Configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html) for more details.

## Usage

Before you start, ensure you have your AWS credentials and OpenAI API key set up correctly.

1. **Setting Environment Variables**
For security reasons, it's recommended to use environment variables to store sensitive information like your AWS access key, secret key, and OpenAI API key.
```bash
export AWS_ACCESS_KEY_ID='your_access_key'
export AWS_SECRET_ACCESS_KEY='your_secret_key'
export OPENAI_API_KEY='your_openai_api_key'
```

2. **Running the Application**
```bash
python main.py
```
Follow the on-screen prompts to enter your query or upload an image.


## Security

Handling secret keys securely is vital. Never hard-code your keys in your scripts or commit them to your repository. Always use environment variables or secure vaults to store them.

For more information on AWS security best practices, please refer to the [AWS Security Best Practices](https://aws.amazon.com/architecture/well-architected/security/).

## License

Distributed under the MIT License. See `LICENSE` for more information.
