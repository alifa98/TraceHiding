# unlearning_experiments


## Clean Up

```bash
# Remove all checkpoints
find experiments -depth -type d -name '*checkpoints*' -exec rm -rf {} +

```


# GPU clean up
```bash
sudo fuser -k /dev/nvidia5
```