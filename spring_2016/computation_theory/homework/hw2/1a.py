def div3(string):
    a = ""
    n = ""
    for char in string:
        n += char
        d, r = divmod(int(n), 3)
        a += str(d)
        n = str(r)
    if r == 0:
        print "accept",
    else:
        print "reject",

def dfa(string):
    state = "s"
    for c in string:
        if state == "s":
            if c == "0":
                state = "r0"
            elif c == "1":
                state = "r1"
            elif c == "2":
                state = "r2"
        elif state == "r0":
            if c == "0":
                state = "r0"
            elif c == "1":
                state = "r1"
            elif c == "2":
                state = "r2"
        elif state == "r1":
            if c == "0":
                state = "r1"
            elif c == "1":
                state = "r2"
            elif c == "2":
                state = "r0"
        elif state == "r2":
            if c == "0":
                state = "r2"
            elif c == "1":
                state = "r0"
            elif c == "2":
                state = "r1"
    if state == "r0":
        print "accept"
    else:
        print "reject"
            


if __name__=="__main__":
    lang = []
    sigma = ["0", "1", "2"]
    for char1 in sigma:
        for char2 in sigma:
            n = char1+char2
            div3(n)
            dfa(n)
