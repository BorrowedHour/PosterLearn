import csv

def processPosters():
    outputFile = open("genre.txt", "wb")
    f = open('./data/MovieGenre.csv', mode="r")
    reader = csv.reader(f)
    reader.next()
    genreSet = set()
    for row in reader:
        gList = row[4].split("|")
        for g in gList:
            if g != "":
                genreSet.add(g)
    for g in genreSet:
        outputFile.write(g)
        outputFile.write("\n")
    outputFile.close()
    f.close()

processPosters()
