from SPARQLWrapper import SPARQLWrapper
import xml.etree.ElementTree as ET

query = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?item
WHERE {
    ?item wdt:P31/wdt:P279* wd:Q124711484
}
OFFSET 6000000
LIMIT  2000000
"""

# OFFSET 10000000  - not successful query for ?item wdt:P31/wdt:P279* wd:Q215627
# LIMIT  1000000

# OFFSET 5700000 - not successful query for ?item wdt:P31/wdt:P279* wd:Q124711484
# LIMIT  100000

# 70681 - Distinct fictional persons count
# 11708091 - Distinct humans (not persons) (Q5) count
# 6739717 - Distinct locations count
# 6427831 - Distinct organizations count

# Q97498056 - fictional person
# Q5 - human
# Q215627 - person
# Q124711484 - spacial region
# Q43229 - organization

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setQuery(query)
sparql.setReturnFormat("xml")
sparql_query = sparql.query()
results = sparql_query.convert()

# Parse XML and extract QIDs
root = ET.fromstring(results.toxml())
numbers = []
for uri in root.findall(".//{http://www.w3.org/2005/sparql-results#}uri"):
    qid = uri.text.split("/")[-1].replace("Q", "")
    numbers.append(qid)

print(f"Writing {len(numbers)} values.")

# Save to file
with open("LOC-ND.txt", "a") as f:
    f.write("\n".join(numbers))
    f.write("\n")

print("Results saved to file")
