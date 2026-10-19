#!/bin/sh
# Print run status as EVT lines, from files on disk (counts only, never claim content).
cd "${1:-.}" || { echo "EVT DIR_MISSING"; exit 0; }
line="EVT"
for d in chunks/*/; do
  lab=$(basename "$d")
  total=$(ls "chunks/$lab" | wc -l)
  done_n=$(ls "stage1/$lab" 2>/dev/null | wc -l)
  if [ -f "stage2/$lab.md" ]; then st=done
  elif [ -d "stage1/$lab" ] && [ "$done_n" -ge "$total" ]; then st=stage2
  elif [ -d "stage1/$lab" ]; then st=stage1
  else st=queued; fi
  line="$line $lab:$st $done_n/$total ;"
done
errs=$(cat errors.log 2>/dev/null | wc -l)
echo "$line errors=$errs"
exit 0
