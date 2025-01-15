import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()



class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0,groq_api_key=os.getenv('GROQ_API_KEY'), model="llama-3.1-70b-versatile")
    
    def extract_jobs(self, page_data):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTIONS:
            The scrapred text is from career's page of a website.
            Your job is to extract the job postings and return a JSON format containing following keys:
            'role', 'skills', 'experience' and 'description'.
            Only return the valid JSON. 
            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input = {'page_data': page_data})

        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse the jobs")
        return res if isinstance(res, list) else [res]
    
    def write_email(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            
            ### INSTRUCTION:
            You are Anas Hussain, a business development executive at Folio3. Folio3 is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Folio3 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Folio3's portfolio: {link_list}
            Remember you are Anas Hussain, BDE at Folio3 bearing the email address "ahussain@folio3.com". 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content


        
