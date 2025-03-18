# Tên index
index_name = "bgl"

config = {
    "template" : {
        "settings" : {
          "index" : {
            "number_of_shards" : "3",
            "number_of_replicas" : "0",
            "routing" : {
              "allocation" : {
                "include" : {
                  "_tier_preference" : "data_content"
                }
              }
            }
          }
        },
        "mappings": {
            "dynamic": False,
            "properties": {
                "LineId": { "type": "integer" },
                "Label": { "type": "keyword" },
                "Timestamp": { "type": "long" },
                "Date": { "type": "date", "format": "yyyy.MM.dd" },
                "Node": { "type": "keyword" },
                "Time": { "type": "date", "format": "yyyy-MM-dd-HH.mm.ss.SSSSSS" },
                "NodeRepeat": { "type": "keyword" },
                "Type": { "type": "keyword" },
                "Component": { "type": "keyword" },
                "Level": { "type": "keyword" },
                "Content": { "type": "text" }
            }
        },
        "aliases" : { }
  }
}
