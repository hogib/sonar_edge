#!/bin/sh
# Fetch the DANS ARIS files listed in dans_list.txt ("id name") into DEST; temp name, rename on success.
LIST="$1"; DEST="$2"
cd "$DEST" || exit 1
while read id name; do
  [ -s "$name" ] && continue
  if curl -sfL --retry 3 -o "$name.part" "https://phys-techsciences.datastations.nl/api/access/datafile/$id"; then
    command mv "$name.part" "$name" && echo "ok $name"
  else
    command rm -f "$name.part"; echo "FAIL $name"
  fi
done < "$LIST"
echo DONE
