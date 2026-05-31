import synapseclient
syn = synapseclient.Synapse()
syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc3OTU5MjM2NCwiaWF0IjoxNzc5NTkyMzY0LCJqdGkiOiIzODI2NiIsInN1YiI6IjM1NDc1MDAifQ.VToAhR5G_1IyoRYUs4UasCBpHRoG9yTyFiio_7WhLH3j_27-_vfZqhxyOp_ZORe9sEmcdLbYc2bvjy4NrLiokuk222Ohb6-oArk1uycPmeNMhQtb-94IVUUVwB5kBCVLCjaP7QSRKdYQm9N6zjCVbNu-47TOngvicwA3pPEEbH3L9Y6PFA6xdicJO9_t_QD1Ng4FDYdq3WFtGJ1gmgchmnFBU-r9pNXOx7Xenr9zcJvWH4HR2rmXZyW7LnrE402EdnOtytEl7hmHgxY-rRI0PMPaziLW4clohWpsrTZra3OX4DNO-WmPLRICYEPVyvDJ9LtGMJFeFareeHDpE92TcQ")

for folder_id, name in [("syn64871173", "segmentations"),
                         ("syn64889927", "patient_info_files")]:
    print(f"\n--- {name} ---")
    for c in syn.getChildren(folder_id):
        print(c["id"], c["name"])