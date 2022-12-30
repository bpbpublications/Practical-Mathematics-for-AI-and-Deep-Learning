import numpy as np
import io
import unicodedata
import re
from tqdm import tqdm

file_path = "./fra-eng/fra.txt"
#Opening and reading the .txt file that contains english and its corresponding italian translation
lines = io.open(file_path, encoding = 'UTF-8').read().split('\n')


def unicode_to_ascii(s) :
  """
  Unicode to ascii conversion
  """
  return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def cleanhtml(raw_html) :
    """
      Function to clean html tags and numbers
      """
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def cleanString(incomingString):
    """
      Function to clean unwanted symbol from text
    """
    newstring = incomingString
    newstring = newstring.replace("!","")
    newstring = newstring.replace("@","")
    newstring = newstring.replace("#","")
    newstring = newstring.replace("$","")
    newstring = newstring.replace("%","")
    newstring = newstring.replace("^","")
    newstring = newstring.replace("&","and")
    newstring = newstring.replace("*","")
    newstring = newstring.replace("(","")
    newstring = newstring.replace(")","")
    newstring = newstring.replace("+","")
    newstring = newstring.replace("=","")
    newstring = newstring.replace("?","")
    newstring = newstring.replace("\'","")
    newstring = newstring.replace("\"","")
    newstring = newstring.replace("{","")
    newstring = newstring.replace("}","")
    newstring = newstring.replace("[","")
    newstring = newstring.replace("]","")
    newstring = newstring.replace("<","")
    newstring = newstring.replace(">","")
    newstring = newstring.replace("~","")
    newstring = newstring.replace("`","")
    newstring = newstring.replace(":","")
    newstring = newstring.replace(";","")
    newstring = newstring.replace("|","")
    newstring = newstring.replace("\\","")
    newstring = newstring.replace("/","")     
    return ' '.join(newstring.split())

def preprocess_string(data) :
    """
      This function calls other
      preprocessing function for
      cleaning data
    """
    data = unicode_to_ascii(data)
    #Remove html
    data = cleanhtml(data)
    #Remove unwanted symbols
    data = cleanString(data)
    return data

def get_data(raw_lines):
    french = []
    english = []
    for itr in tqdm(range(len(raw_lines))):
        if len(raw_lines[itr].split()) > 2:
            eng, fre, _ = raw_lines[itr].split('\t')
            english.append('<start> ' + preprocess_string(eng) + ' <end>')
            french.append('<start> ' + preprocess_string(fre) + ' <end>')
    return french, english