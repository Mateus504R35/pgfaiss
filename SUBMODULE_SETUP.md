# Adding Faiss v1.13.1 as a Git submodule

Run these commands from the root of the new pgfaiss Git repository:

```bash
rm -f third_party/.gitkeep
git submodule add https://github.com/facebookresearch/faiss.git third_party/faiss
git -C third_party/faiss checkout v1.13.1
git add .gitmodules third_party/faiss
git commit -m "Add Faiss v1.13.1 as submodule"
```

After that, verify the pinned revision:

```bash
git -C third_party/faiss describe --tags --exact-match
git submodule status
```

The first command should print:

```text
v1.13.1
```

Future clones should use:

```bash
git clone --recurse-submodules <YOUR_REPOSITORY_URL>
```

or, after a normal clone:

```bash
git submodule update --init --recursive
```
