import torch
## 提前取数据加速训练速度
class DataFetcher:
    def __init__(self, torch_loader):
        self.torch_loader = torch_loader
    def __len__(self):
        return len(self.torch_loader)

    def __iter__(self):
        self.stream = torch.cuda.Stream()
        self.loader = iter(self.torch_loader)
        self.preload()
        return self

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_depth_map, self.next_label = sample['image_x'], sample['map_x'], sample['spoofing_label']
        except StopIteration:
            self.next_input = None
            self.next_depth_map = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_depth_map = self.next_depth_map.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        depth_map = self.next_depth_map
        label = self.next_label

        if input is not None:
            input.record_stream(torch.cuda.current_stream())
            depth_map.record_stream(torch.cuda.current_stream())
            label.record_stream(torch.cuda.current_stream())
        else:
            raise StopIteration
        self.preload()
        return {'image_x': input, 'map_x': depth_map, 'spoofing_label': label}

if __name__ == '__main__':
    import torch
    a = torch.randn(2, 2, 2)
    print(a)
    mean = torch.mean(a, axis=(1, 2))
    print(mean.shape)
    print(mean)