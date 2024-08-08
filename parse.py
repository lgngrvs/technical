from markdown import markdown
import os 

parent_directory = "markdown/"
list_of_file_names = os.listdir(parent_directory)
print(list_of_file_names)

for file_name in list_of_file_names: 
    file_path = f'{parent_directory}{file_name}'
    content = ""

    with open(file_path) as file: # opens the markdown file
        file_text = file.read()
        content = markdown(file_text, extensions=["tables", "footnotes"])
    
    print(content)

    with open(file_name[:-3] + ".html", "w") as file: # opens a new html file with the filename
        print(content)
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
        </body>
        </html>
        """
        file.write(completed_template)