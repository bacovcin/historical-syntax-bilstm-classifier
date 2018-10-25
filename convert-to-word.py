import sys
from PTree import *
from random import shuffle
import progressbar

DOCODES = {'DO': '1',
           'MOD': '2',
           'NOTDO': '3',
           'SUBQ': '1',
           'SBJQ': '2',
           'QUE': '3',
           'IMP': '4',
           'DEC': '5',
           'OTHER': '6',
           'NEG': '1',
           'AFF': '2',
           '_': '0'}

def extract_token_id(tree):
    output = ''
    if tree.name == 'ID':
        output = tree.content
    elif tree.height > 0:
        for child in tree.content:
            output += extract_token_id(child)
    return output


def write_out_diff(t1, t2, outfile, numout=False):
    temp1 = open('temp1.txt', 'w', encoding='utf8')
    temp1.write(str(t1))
    temp1.close()
    temp2 = open('temp2.txt', 'w', encoding='utf8')
    temp2.write(str(t2))
    temp2.close()
    output = (subprocess.run(['diff', 'temp1.txt',
                              'temp2.txt'], stdout=subprocess.PIPE).stdout)
    if numout == 2:
        if '(NUM' in str(output):
            tid = extract_token_id(t1)
            outfile.write(bytes(tid+'\n', 'UTF-8'))
            outfile.write(output)
            outfile.write(bytes('\n', 'UTF-8'))
    elif numout == 1:
        if ('(N means' not in str(output) and
            '(N Means' not in str(output)):
            tid = extract_token_id(t1)
            outfile.write(bytes(tid+'\n', 'UTF-8'))
            outfile.write(output)
            outfile.write(bytes('\n', 'UTF-8'))
    elif ('(NUM' not in str(output) and
          '(N means' not in str(output) and
          '(N Means' not in str(output)):
        tid = extract_token_id(t1)
        outfile.write(bytes(tid+'\n', 'UTF-8'))
        outfile.write(output)
        outfile.write(bytes('\n', 'UTF-8'))
    subprocess.run(['rm', 'temp1.txt'])
    subprocess.run(['rm', 'temp2.txt'])
    return


def process_file_pair(f1, f2, debug, numout):
    trees = ParseFiles([f1, f2], debug)
    key1 = f1.split('.')[0]
    key2 = f2.split('.')[0]

    outfile = open('diff.txt','ab')
    outfile.write(bytes(key1+'\n'+key2+'\n-------\n','UTF-8'))

    for i in range(min((len(trees[key1]),len(trees[key2])))):
        if debug:
            print(trees[key1][i])
        t1 = delemmatize(trees[key1][i],debug)
        if debug:
            print(trees[key2][i])
        t2 = delemmatize(trees[key2][i],debug)
        if t1 != t2:
            write_out_diff(t1, t2, outfile, numout)


def get_words(tree, worddict, code):
    words = []
    if type(tree) is str:
        pass
    elif tree.height == 0 and tree.name == 'CODING-CP':
        code = [DOCODES[x] for x in tree.content.split(':')]
    elif 'ORTHO' in [x.name for x in tree.content if type(x) is not str]:
        form = [x.content.lower() for x in tree.content
                if x.name == 'ORTHO'][0]
        try:
            wordid = worddict[form]
        except KeyError:
            wordid = str(max([int(x) for x in worddict.values()]) + 1)
        worddict[form] = wordid
        if tree.name[0] == 'V' and tree.name != 'VAG':
            words.append([wordid] + code)
            code = ['0', '0', '0']
        else:
            words.append([wordid, '0', '0', '0'])
    else:
        for child in tree.content:
            newwords, worddict, code = get_words(child, worddict, code)
            words += newwords
    return words, worddict, code


def write_out_words(tokens, filename, worddict):
    with progressbar.ProgressBar(max_value=len(tokens)) as bar:
        with open(filename, 'w') as outfile:
            for i, token in enumerate(tokens):
                words, worddict, code = get_words(token, worddict, ['0', '0', '0'])
                for word in words:
                    outfile.write(word[0] + '\t' +
                                  '\t'.join(word[1:]) + '\n')
                outfile.write('\n')
                bar.update(i)
    return worddict


def main():
    worddict = {}
    with progressbar.ProgressBar(max_value=2196017) as bar:
        with open('./glove.840B.300d.txt') as infile:
            for i, line in enumerate(infile):
                sline = line.split()
                worddict[sline[0]] = str(i)
                bar.update(i)
    tokens = ParseFile(sys.argv[-1])
    shuffle(tokens)
    training = tokens[:int(len(tokens)*0.8)]
    dev = tokens[int(len(tokens)*0.8):int(len(tokens)*0.9)]
    test = tokens[int(len(tokens)*0.9):]
    worddict = write_out_words(training, 'training.conll', worddict)
    worddict = write_out_words(dev, 'dev.conll', worddict)
    worddict = write_out_words(test, 'test.conll', worddict)
    with open('worddict', 'w') as outfile:
        for word in worddict:
            outfile.write(word + '\t' + worddict[word] + '\n')


if __name__ == '__main__':
    main()
