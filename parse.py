from markdown import markdown
import os 

parent_directory = "markdown/"
list_of_file_names = os.listdir(parent_directory)
print(list_of_file_names)


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


for file_name in list_of_file_names: 
    file_path = f'{parent_directory}{file_name}'
    content = ""

    with open(file_path) as file: # opens the markdown file
        file_text = file.read()
        content = markdown(file_text, extensions=["tables", "footnotes"])
    
    print(content)

    with open(file_name[:-3] + ".html", "w") as file: # opens a new html file with the filename
        print(content)
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

        file.write(completed_template)