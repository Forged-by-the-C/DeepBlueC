import argparse
import sys

try:
    from sklearn.metrics import f1_score
except:
    print("[-] Missing sklearn.metrics module.")
    print("[+] pip install -U scikit-learn")
    sys.exit(4)

    
################################## Func ##################################


def getfscore(subFile, trainFile):
    submission, key = openFiles(subFile, trainFile)

    keymap = create_map(key)
    submap = create_map(submission)
    
    if keymap == None or submap == None:
        sys.stderr.write("[-] Invalid format in input file(s).\n")
        sys.exit(3)
        
    f1 = fscore(submap, keymap)

    return f1

# Slow, but works on data set
def fscore(submission, key):
    guess_score = []
    true_score = []

    it = iter(sorted(key.iteritems()))
    tup = it.next()
    while tup:
        true_score.append(tup[1])
        guess_score.append(submission[tup[0]])
        try:
            tup = it.next()
        except:
            break
    
    return f1_score(true_score, guess_score, average='micro')

def create_map(filehandle):
    header = filehandle.readline()
    header = header.split(",")
    if header[0] != "building_id" or header[1].rstrip() != "damage_grade":
        return None

    resultMap = {}
    for line in filehandle:
        group = line.split(",")
        try:
            resultMap[int(group[0])] = int(group[1].rstrip())
        except:
            sys.stderr.write("[-] Error: Failure to insert ({},{}) to map\n".format(group[0],group[1].rstrip()))
            return None

    return resultMap
            
def openFiles(sub, train):
    try:
        submit_file = open(sub)
    except:
        sys.stderr.write("[-] Error: Failure to open {}\n".format(sub))
        sys.exit(1)

    try:
        training_file = open(train)
    except:
        sys.stderr.write("[-] Error: Failure to open {}\n".format(train))
        sys.exit(1)
        
    return submit_file, training_file


################################## Main ##################################

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Score given .csv file relative to provided training labels file')
    parser.add_argument('submit', metavar='submit_csv', type=str,
                   help='.csv file for scoring ')
    parser.add_argument('training', metavar='training_csv', type=str, help='.csv file to compare (train_labels.csv)')

    args = parser.parse_args()

    f1 = getfscore(args.submit, args.training)
    
    print("[+] F1 score of submission is: {}".format(f1))

