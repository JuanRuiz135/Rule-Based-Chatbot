#######################################################
# Initializer first order logic based agent
#######################################################
import nltk
v = """
Boran => br
Berneses => {}
Huskies => {}
Sheperds => {}
Retrievers => {}
Pomskies => {}
Bernese => {}
Husky => {}
Sheperd => {}
Retriever => {}
Pomsky => {}
Barkers => dr1
Doggoland => dr2
Puppies => dr3
be_in => {}
are_in => {}
is_in => {}
is_named => {}
"""
folval = nltk.Valuation.fromstring(v)
grammar_file = 'simple-sem.fcfg'

objCounter = 0
# Inserting knowledge
def sheltersKnowledge():
    objectCounter = 0
    dogbreedNum= 0
    o = 'o' + str(objectCounter)
    objectCounter += 1
    #insert type of dog 
    folval["Pomskies"].add((o,))
    folval["Berneses"].add((o,))
    folval["Sheperds"].add((o,))
    folval["Retrievers"].add((o,))
    folval["Huskies"].add((o,))
    folval["Pomsky"].add((o,))
    folval["Bernese"].add((o,))
    folval["Sheperd"].add((o,))
    folval["Retriever"].add((o,))
    folval["Husky"].add((o,))  
    if len(folval["be_in"]) == 1: #clean up if necessary
        if ('',) in folval["be_in"]:
            folval["be_in"].clear()
    if dogbreedNum != 10:
        dogbreedNum = dogbreedNum + 1 
        folval["be_in"].add((o, folval["Barkers"])) #insert location
        folval["be_in"].add((o, folval["Doggoland"])) 
        folval["be_in"].add((o, folval["Puppies"])) 