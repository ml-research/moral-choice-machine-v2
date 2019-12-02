# python data/alto_tools/alto_tools.py /home/patrick/repositories/MoralChoiceMachine/Code_python3.5/data/ALTO/000000874_000013.xml -t > /home/patrick/repositories/MoralChoiceMachine/Code_python3.5/data/ALTO/000000874_000013.txt
from tqdm import tqdm
import os
from data.alto_tools.alto_tools import alto_parse, alto_text
#

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created dir:", directory)

#path_base = 'data/books/XML_FILES/1510_1699'
#path_base = 'data/books/XML_FILES/1700_1799'
#path_base = 'data/books/XML_FILES/1800_1809'
#path_base = 'data/books/XML_FILES/1890_1899'
#path_base = 'data/books/XML_FILES/1810_1819'
#path_base = 'data/books/XML_FILES/1820_1829'

#path_base = 'data/books/XML_FILES/1840_1849'
#path_base = 'data/books/XML_FILES/1880_1889'
path_base = 'data/books/XML_FILES/1860_1869'
#path_base = 'data/books/XML_FILES/1870_1879'

last_book_id = None
book_text = list()

for root, dirs, files in os.walk(path_base):
    for file in files:
        if 'ALTO' in root and file.endswith(".xml"):
            path = os.path.join(root, file)
            file_name = file[:-4]
            file_id = file_name.split("_")

            # first write book to disk if
            if last_book_id is None:
                last_book_id = file_id
            elif last_book_id[0] != file_id[0] and len(book_text) > 0:
                # save last_book_id
                save_path = path.split("/")[:4]
                save_path = "/".join(save_path)
                file_path = os.path.join(save_path, 'parsed_data')
                file_path = os.path.join(file_path, last_book_id[0] + '.txt')

                if os.path.exists(file_path) and os.path.isfile(file_path):
                    raise ValueError("file already exists")

                ensure_dir(file_path)
                book_text.sort(key=lambda x: x[2])
                book_text_string = ''
                for text_list, _, _ in book_text:
                    text_string = ''
                    for text in text_list:
                        text_string += text
                    book_text_string += text_string

                save_file = open(file_path, "w")
                save_file.write(book_text_string)
                save_file.close()
                print("saved book:", file_path)
                book_text = list()

            save_path = path.split("/")[:4]
            save_path = "/".join(save_path)
            file_path = os.path.join(save_path, 'parsed_data')
            file_path = os.path.join(file_path, file_id[0] + '.txt')

            if not (os.path.exists(file_path) and os.path.isfile(file_path)):
                alto = open(path, 'r', encoding='UTF8')
                try:
                    alto, xml, xmlns = alto_parse(alto)
                    book_text.append((alto_text(xml, xmlns), file_id[0], int(file_id[1])))
                    last_book_id = file_id
                except UnboundLocalError:
                    pass
            else:
                print("book already parsed:", file_path)





