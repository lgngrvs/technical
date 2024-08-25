from markdown import markdown
import os 
import pygments
import re

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from pygments.lexers import guess_lexer

import html

parent_directory = "markdown/"
list_of_file_names = os.listdir(parent_directory)
# print(list_of_file_names)


def generated_index(): 
    thing_to_append_to_content = ""

    bullets = ""

    for file_name in list_of_file_names: 
        file_path = f'{parent_directory}{file_name}'

        link = file_name[:-3] + ".html"
        linktext = ""

        with open(file_path) as file: 
            linktext += remove_non_alphanumeric_keep_spaces(file.readline())
        
        bullet_template = f"""<li><a href="{link}">{linktext}</a></li>"""
        if file_name != "index.md": 
            bullets += bullet_template
                
    list_template = f"""
        <ul>
            {bullets}
        </ul>
        """

    return list_template

def remove_non_alphanumeric_keep_spaces(string):
	return ''.join(char for char in string if char.isalnum() or char.isspace() or char=="-")

# def add_code_class(html_string):
#	return html_string.replace("<code", "<code class='microlight'")


for file_name in list_of_file_names:
    file_path = f'{parent_directory}{file_name}'
    content = ""

    with open(file_path) as file: # opens the markdown file
        file_text = file.read()
        content = markdown(file_text, extensions=["tables", "footnotes"])

    # print(content)

    with open(file_name[:-3] + ".html", "w") as file: # opens a new html file with the filename
        #print(content)
        if file_name == "index.md": 
            content += generated_index()

        completed_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>new life</title>
            <link rel="stylesheet" href="site.css">
            <link rel="stylesheet" href="theme.css">
        </head>
        <body>
            <div id="wrapper">
                {content}
            </div>
            <script type="text/javascript" src="mathjax.js"></script>
            <script type="text/javascript" id="MathJax-script" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

        </body>
        </html>
        """

        content_formatted = completed_template

        for code_section in re.findall('\<pre\>\<code\>[\s\S]*?\<\/code\>\<\/pre\>', content_formatted):
	        # print(code_section)
                new_code_section = code_section.replace('<pre><code>', '')
                new_code_section = new_code_section.replace('</code></pre>', '')

                new_code_section = html.unescape(new_code_section)

                lexer = get_lexer_by_name("python", stripall=True)
                formatter = HtmlFormatter(linenos=False, cssclass="bw", style='bw')

                new_code_section_highlight = highlight(new_code_section, lexer, formatter)
                content_formatted = content_formatted.replace(code_section, new_code_section_highlight)
        # print(content_formatted)
        # completed_template = add_code_class(completed_template)
        file.write(content_formatted)

css = HtmlFormatter(style="bw").get_style_defs('.bw')

with open("theme.css", "w") as file:
	file.write(css)
