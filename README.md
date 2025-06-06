# Health Research MCP System

This Multi-Agent System (MCP) is designed to research health topics and generate comprehensive health articles. The system uses multiple specialized agents to gather information, analyze findings, and produce well-structured articles.

## Features

- Research Agent: Gathers latest developments and scientific findings
- Analysis Agent: Evaluates and summarizes research data
- Writing Agent: Generates comprehensive health articles

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the main script with Python:
```bash
python health_mcp.py
```

By default, it will generate an article about recent developments in CRISPR gene editing. To modify the topic, edit the `topic` variable in the `main()` function of `health_mcp.py`.

## Output

The system generates:
1. A JSON file containing the research data, analysis, and final article
2. A printed version of the generated article in the console

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for research

