import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
import json

# Load environment variables
load_dotenv()

class HealthMCP:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def research_agent(self, topic: str) -> Dict:
        """Research agent that gathers information about health topics"""
        prompt = f"""Research the latest developments in {topic}. 
        Focus on recent studies, breakthroughs, and scientific consensus.
        Format the response as a JSON with keys: 'findings', 'sources', 'key_points'"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        return json.loads(response.choices[0].message.content)
    
    def analysis_agent(self, research_data: Dict) -> Dict:
        """Analysis agent that evaluates and summarizes research findings"""
        prompt = f"""Analyze the following health research data and provide insights:
        {json.dumps(research_data, indent=2)}
        Format the response as a JSON with keys: 'summary', 'implications', 'recommendations'"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        return json.loads(response.choices[0].message.content)
    
    def writing_agent(self, analysis_data: Dict, research_data: Dict) -> str:
        """Writing agent that generates a health article"""
        prompt = f"""Write a comprehensive health article based on the following research and analysis:
        Research: {json.dumps(research_data, indent=2)}
        Analysis: {json.dumps(analysis_data, indent=2)}
        
        The article should:
        1. Have a clear introduction
        2. Present the latest findings
        3. Explain implications for public health
        4. Include practical recommendations
        5. Be written in a clear, engaging style
        6. Be suitable for a general audience while maintaining scientific accuracy"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def generate_health_article(self, topic: str) -> Dict:
        """Main function to generate a health article"""
        # Step 1: Research
        research_data = self.research_agent(topic)
        
        # Step 2: Analysis
        analysis_data = self.analysis_agent(research_data)
        
        # Step 3: Writing
        article = self.writing_agent(analysis_data, research_data)
        
        return {
            "topic": topic,
            "research": research_data,
            "analysis": analysis_data,
            "article": article
        }

def main():
    # Example usage
    mcp = HealthMCP()
    topic = "recent developments in CRISPR gene editing for treating genetic diseases"
    result = mcp.generate_health_article(topic)
    
    # Save the results
    with open(f"health_article_{topic.replace(' ', '_')}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    # Print the article
    print("\nGenerated Article:")
    print("=" * 80)
    print(result["article"])
    print("=" * 80)

if __name__ == "__main__":
    main() 