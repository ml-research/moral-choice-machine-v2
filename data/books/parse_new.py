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


paths = list()

#paths.append('/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1510_1699')
#paths.append('/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1700_1799')
#paths.append('/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1800_1809')
#paths.append('/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1810_1819')
#paths.append('/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1820_1829')
#paths.append('/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1830_1839')
paths.append('/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1840_1849')
paths.append('/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1850_1859')
paths.append('/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1860_1869')
paths.append('/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1870_1879')
paths.append('/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1880_1889')
#paths.append('/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1890_1899')

for path_base in paths:
    last_book_id = None

    # collect data
    files_dict = dict()
    for root, dirs, files in tqdm(os.walk(path_base)):
        for file in files:
            if 'ALTO' in root and file.endswith(".xml"):
                path = os.path.join(root, file)
                file_name = file[:-4]
                file_id = file_name.split("_")

                if file_id[0] not in files_dict:
                    files_dict[file_id[0]] = list()
                files_dict[file_id[0]].append((path, int(file_id[1])))

    print("Finished reading files")
    books_keys = list(files_dict.keys())
    print("Writing {} files", len(books_keys))
    for idx, book_key in tqdm(enumerate(books_keys)):
        book_parts = files_dict[book_key]
        book_parts.sort(key=lambda x: x[1])
        book_text_string = ''
        for file_path, book_part_id in book_parts:
            alto = open(file_path, 'r', encoding='UTF8')
            alto, xml, xmlns = alto_parse(alto)
            xml_text = alto_text(xml, xmlns)
            text_string = ''
            for text in xml_text:
                text_string += text
            book_text_string += text_string
        #if idx == 200:
        #    break
        save_path = path_base
        file_path = os.path.join(save_path, 'parsed_data')
        file_path = os.path.join(file_path, book_key + '.txt')
        ensure_dir(file_path)
        save_file = open(file_path, "w")
        save_file.write(book_text_string)
        save_file.close()







