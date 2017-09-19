import csv
from PIL import Image

genreMap = {}
genreMap['Sci-Fi'] = 0
genreMap['Crime'] = 1
genreMap['Romance'] = 2
genreMap['Animation'] = 3
genreMap['Music'] = 4
genreMap['Adult'] = 5
genreMap['Comedy'] = 6
genreMap['War'] = 7
genreMap['Horror'] = 8
genreMap['Film-Noir'] = 9
genreMap['Adventure'] = 10
genreMap['News'] = 11
genreMap['Reality-TV'] = 12
genreMap['Thriller'] = 13
genreMap['Western'] = 14
genreMap['Mystery'] = 15
genreMap['Short'] = 16
genreMap['Talk-Show'] = 17
genreMap['Drama'] = 18
genreMap['Action'] = 19
genreMap['Documentary'] = 20
genreMap['Musical'] = 21
genreMap['History'] = 22
genreMap['Family'] = 23
genreMap['Fantasy'] = 24
genreMap['Game-Show'] = 25
genreMap['Sport'] = 26
genreMap['Biography'] = 27

def normalize(pixel):
    p = pixel / 255.0
    return str(round(p,1))

def processPoster(count, name, posterId, genres, file):
    try:
        im = Image.open('./data/posters/' + posterId + '.jpg', 'r')
        pix = im.load()
        x = im.size[0]
        y = im.size[1]
        if (x == 182 and y == 268):
            print "Processing : " + str(count) + " : " +name
            genreList = genres.split("|")
            for g in genreList:
                file.write("\n")
                file.write(str(genreMap[g]) + ",")
                for a in range(x):
                    for b in range(y):
                        file.write(normalize(pix[a,b][0]) + "," + normalize(pix[a,b][1]) + "," + normalize(pix[a,b][2]) + ",")

            file.flush()
        else:
            print "Ignoring " + name + " size not the same.."
    except:
        print "Error processing : " + name

def processPosters():
    outputFile = open("output.csv", "wb")
    f = open('./data/MovieGenre.csv', mode="r")
    reader = csv.reader(f)
    reader.next()
    count = 0
    for row in reader:
        count = count + 1
        processPoster(count, row[2], row[0], row[4], outputFile)
        if count == 1000:
            break
    f.close()
    outputFile.close()


processPosters()
