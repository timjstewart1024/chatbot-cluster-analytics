import utilities
import csv


df = utilities.combine_all_conversations()

df.to_csv('all_conversations.csv', index=False, quoting=csv.QUOTE_STRINGS)
