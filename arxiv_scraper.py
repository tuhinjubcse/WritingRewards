from bs4 import BeautifulSoup
import requests

# Define block-level elements
BLOCK_ELEMENTS = {
    'p', 'div', 'section', 'article', 'aside', 'nav', 'header', 'footer', 
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'blockquote',
    'pre', 'table', 'tr', 'td', 'th', 'figure', 'figcaption', "ul", "li"
}

def get_text_with_block_newlines(element):
    """Extract text with newlines only for block-level elements."""
    text_parts = []
    
    for child in element.children:
        if isinstance(child, str):
            # Handle text nodes
            text_parts.append(child.strip())
        else:
            # Handle element nodes
            child_text = child.get_text(separator=' ', strip=True)
            if child_text:
                text_parts.append(child_text)
                if child.name in BLOCK_ELEMENTS:
                    text_parts.append('\n')
    
    return ' '.join(text_parts).strip()

def extract_paper_content(arxiv_url):
    response = requests.get(arxiv_url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    content = soup.find('div', {'class': 'ltx_page_content'})
    
    if not content:
        raise ValueError("Could not find paper content in the HTML")
    
    full_content = get_text_with_block_newlines(content)
    
    # Find all sections
    sections = content.find_all('section', {'class': 'ltx_section'})
    
    if not sections:
        raise ValueError("Could not find any sections in the paper")
    
    # The first section should be the introduction
    intro_section = sections[0]
    intro_content = get_text_with_block_newlines(intro_section)
    
    return full_content, intro_content

# Example usage
if __name__ == "__main__":
    url = "https://arxiv.org/html/2309.14556v3"
    try:
        full_content, intro_content = extract_paper_content(url)
        print("Full content length:", len(full_content))
        print("Content up to introduction length:", len(intro_content))
    except Exception as e:
        print(f"Error: {e}") 
