import httpx
from bs4 import BeautifulSoup
from urllib.parse import quote
import asyncio
from PyPDF2 import PdfReader
from io import BytesIO
from docx import Document
from copy import deepcopy
from src.validator import validator_online

async def fetch_page_text(url: str, session: httpx.AsyncClient) -> str:
    """Fetches the text content of a given URL, retrying twice if necessary."""
    try:
        response = await session.get(url, follow_redirects=True)
        response.raise_for_status()  # Raises an exception for 4XX/5XX responses

        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' in content_type:
            return await fetch_pdf_text(response)
        elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
            return await fetch_docx_text(response)
        else:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
            text = soup.get_text(separator=' ', strip=True)
            if "please click here if the page does not redirect automatically" in text.lower():
                return "Failed to retrieve content"
            return text
    except Exception as e:
        print(f"Failed to fetch page text for {url}: {e}")
        return "Failed to retrieve content"

async def fetch_pdf_text(response: httpx.Response) -> str:
    """Extracts text from a PDF document."""
    try:
        reader = PdfReader(BytesIO(response.content))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Failed to read PDF: {e}")
        return "Failed to retrieve content"

async def fetch_docx_text(response: httpx.Response) -> str:
    """Extracts text from a DOCX document."""
    try:
        doc = Document(BytesIO(response.content))
        text = " ".join(paragraph.text for paragraph in doc.paragraphs)
        return text
    except Exception as e:
        print(f"Failed to read DOCX: {e}")
        return "Failed to retrieve content"

async def fetch_search_page(url: str, session: httpx.AsyncClient, headers: dict) -> str:
    """Fetches the search result page for the given URL."""
    try:
        response = await session.get(url, headers=headers, follow_redirects=True)
        response.raise_for_status()  # Raises an exception for 4XX/5XX responses
        return response.text
    except Exception as e:
        print(f"Failed to fetch search page: {e}")
        return None

async def extract_google_results(html_content: str, session: httpx.AsyncClient) -> list:
    """Extracts links and snippets from Google search results."""
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    for result in soup.find_all('div', class_='tF2Cxc'):
        title_element = result.find('h3')
        title = title_element.get_text() if title_element else "No title found"
        
        link_element = result.find('a')
        link = link_element['href'] if link_element else "No link found"

        snippet_elem = result.select_one('.VwiC3b')
        snippet = snippet_elem.get_text() if snippet_elem else "No snippet found"

        results.append({'engine': 'Google', 'title': title, 'link': link, 'snippet': snippet})

    return results

async def extract_bing_results(html_content: str, session: httpx.AsyncClient) -> list:
    """Extracts links and snippets from Bing search results."""
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    for result in soup.find_all('li', class_='b_algo'):
        title_element = result.find('h2')
        title = title_element.get_text() if title_element else "No title found"
        
        link_element = result.find('a')
        link = link_element['href'] if link_element else "No link found"

        snippet_elem = result.find('p')
        snippet = snippet_elem.get_text() if snippet_elem else "No snippet found"

        results.append({'engine': 'Bing', 'title': title, 'link': link, 'snippet': snippet})

    return results

async def fetch_page_text_with_retries(url: str, session: httpx.AsyncClient, retries: int = 2) -> str:
    """Fetches the page text with retries."""
    for attempt in range(retries):
        page_text = await fetch_page_text(url, session)
        if page_text != "Failed to retrieve content":
            return page_text
    return "Failed to retrieve content"

async def get_snippet(query: str, engine: str, session: httpx.AsyncClient = None):
    """Performs search and extract links and snippets for the specified search engine."""
    if session is None:
        async with httpx.AsyncClient() as session:
            return await _get_snippet(query, engine, session)
    else:
        return await _get_snippet(query, engine, session)

async def _get_snippet(query: str, engine: str, session: httpx.AsyncClient):
    """Helper function to perform search and extract links and snippets for the specified search engine."""
    encoded_query = quote(query)
    
    if engine == 'google':
        url = f"https://www.google.com/search?q={encoded_query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.7,en-US;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "TE": "trailers",
        }
        html_content = await fetch_search_page(url, session, headers)
        if html_content:
            return await extract_google_results(html_content, session)
    
    elif engine == 'bing':
        url = f"https://www.bing.com/search?q={encoded_query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.7,en-US;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "TE": "trailers",
        }
        html_content = await fetch_search_page(url, session, headers)
        if html_content:
            return await extract_bing_results(html_content, session)
    
    return []

async def fetch_all_snippets(queries: list, question: str, model_name='gpt-3.5', engines=['google'], validator=True, UI=True):
    async with httpx.AsyncClient() as session:
        tasks = [get_snippet(query, engine, session) for query in queries for engine in engines]
        results = await asyncio.gather(*tasks)
        
        combined_results = {}
        for i, query in enumerate(queries):
            combined_results[query] = []
            for j, engine in enumerate(engines):
                combined_results[query].extend(results[i * len(engines) + j])
                
        if validator:
            snippet_list = [item['snippet'] for query, results in combined_results.items() for item in results]
            good_snippets = validator_online(question, snippet_list, model_name=model_name, UI_flag=UI)
            combined_results_copy = deepcopy(combined_results)
            for query, results in combined_results_copy.items():
                combined_results[query] = [item for item in results if item['snippet'] in good_snippets]

        tasks = [fetch_page_text_with_retries(item['link'], session) for query, results in combined_results.items() for item in results if item['link'] != "No link found"]
        pages_text = await asyncio.gather(*tasks)
        
        idx = 0
        for query, results in combined_results.items():
            for item in results:
                item['text'] = pages_text[idx]
                idx += 1
                
        for query in combined_results:
            combined_results[query] = [result for result in combined_results[query] if result['text'] is not None]
                
        return combined_results

# Example usage:
# async def main():
#     queries = [
#         "in which information element the UPF send the data volume per QoS flow to the SMF?",
#         "how to use asyncio in python",
#         "best practices for web scraping"
#     ]
#     engines = ["google", "bing"]
#     snippets = await fetch_all_snippets(queries, engines)
#     for query, results in snippets.items():
#         print(f"Results for query: {query}")
#         for item in results:
#             print(f"Search Engine: {item['engine']}\nTitle: {item['title']}\nLink: {item['link']}\nSnippet: {item['snippet']}\nText: {item['text'][:500]}...\n")  # Displaying first 500 chars for brevity

# if __name__ == "__main__":
#     asyncio.run(main())
