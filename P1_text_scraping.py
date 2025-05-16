import requests
from bs4 import BeautifulSoup

def scrape_wikipedia(topic):
    """
    Scrape text data from Wikipedia for a given topic.
    
    Args:
        topic (str): The topic to search for on Wikipedia
        
    Returns:
        str: The scraped text content
    """
    # Format the topic for URL
    formatted_topic = topic.replace(' ', '_')
    url = f"https://en.wikipedia.org/wiki/{formatted_topic}"
    
    # Send request to Wikipedia
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get the main content
        content_div = soup.find('div', {'id': 'mw-content-text'})
        
        if content_div:
            # Extract paragraphs
            paragraphs = content_div.find_all('p')
            text = '\n\n'.join([p.get_text() for p in paragraphs])
            return text
        else:
            return "Content not found on the page."
    else:
        return f"Failed to retrieve the Wikipedia page. Status code: {response.status_code}"

if __name__ == "__main__":
    # Example usage
    topic = "Natural language processing"
    scraped_text = scrape_wikipedia(topic)
    
    # Print first 500 characters of the scraped text
    print("Scraped text preview:")
    print(scraped_text[:500])
    
    # Save the scraped text to a file
    with open(f"{topic.replace(' ', '_')}_wiki.txt", "w", encoding="utf-8") as file:
        file.write(scraped_text)
    
    print(f"\nScraped text has been saved to {topic.replace(' ', '_')}_wiki.txt") 