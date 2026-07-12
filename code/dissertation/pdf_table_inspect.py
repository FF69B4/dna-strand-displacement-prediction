import pdfplumber

with pdfplumber.open("data/mmc1.pdf") as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables()
        print(f"\nPAGE {page_num}: {len(tables)} table(s)")

        for table_num, table in enumerate(tables, start=1):
            print(f"\nTable {table_num}")
            for row in table[:4]:
                print(row)