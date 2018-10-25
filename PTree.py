import sys

class PTree:
    def __init__(self, name, content):
        if type(name) is str and ' ' not in name:
            self.name = name
        else:
            print("Name is not a string")
        self.height = 0
        if type(content) is list:
            newcontent = [] 
            for tree in content:
                if type(tree) is str:
                    continue
                newcontent.append(tree)
                if tree.height >= self.height:
                    self.height = tree.height + 1
        else:
            newcontent = content
        self.content = newcontent

    def __hash__(self):
        return hash(str(self))

    def __ne__(self,other):
        return not(hash(self)==hash(other))

    def __eq__(self,other):
        return hash(self) == hash(other)

    def __str__(self):
        if (type(self.content) is str):
            output = '\n(' + self.name + ' ' + self.content + ')'
        else:
            output = '\n(' + self.name
            try:
                for y in self.content:
                    text = str(y).split('\n')
                    output = output + '\n  '.join(text)
            except TypeError:
                print(self.name)
                print(self.content)
                sys.exit()
            output = output + '\n)'
        return output


def MatchParen(x):
    output = []
    outtext = ''
    i = 0
    while i < len(x):
        c = x[i]
        if c == '(':
            outtext = ''.join(''.join(outtext.split(' ')).split('\t'))
            if outtext != '':
                output.append(outtext)
            outtext = ''
            y = MatchParen(x[i+1:])
            output.append(y[0])
            i = i+y[1]
        elif c == ')':
            if outtext not in [' ', '', '\t']:
                output.append(outtext)
            break
        else:
            outtext = outtext + c
            i = i + 1
    return (output, i+2)


def ParseTree(x):
    if len(x) > 1 or type(x[0]) is list:
        try:
            name = x[0].rstrip()
            start = 1
        except:
            name = ''
            start = 0
        content = []
        for y in x[start:]:
            if type(y) is list:
                content.append(ParseTree(y))
            else:
                content.append(y)
    else:
        try:
            y = x[0].split(' ')
            name = y[0]
            content = y[1]
        except IndexError:
            print(x)
            raise "We crashed..."
    return PTree(name, content)


def ParseFile(filename, debug=False):
    infile = open(filename, encoding="latin-1")
    tokens = []
    token = ''
    for line in infile:
        if (line == '\n' or
            line == '(\n') and ('ID' in token or 'CODE' in token):
            if debug:
                print(token)
            tokens.append(ParseTree(MatchParen(token.lstrip().rstrip())
                                    [0][0]))
            if line == '\n':
                token = ''
            else:
                token = line.rstrip().lstrip()
        elif (line == '\n'):
            if line == '\n':
                token = ''
            else:
                token = line.rstrip().lstrip()
        else:
            token = token + line.rstrip().lstrip()
    if ('ID' in token or 'CODE' in token):
        tokens.append(ParseTree(MatchParen(token.lstrip().rstrip())[0][0]))
    return tokens


def ParseFiles(argvs,debug=False):
    toklist = {}
    for i in range(len(argvs)):
        arg = argvs[i]
        if arg[-3:] in ['ref', 'psd']:
            toklist[arg[:-4]] = ParseFile(arg)
    return toklist
