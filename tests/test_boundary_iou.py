from test_instance_metrics import create_test_images
from thaw_slump_segmentation.metrics import BinaryBoundaryIoU


def test_boundary_iou():
    preds, target = create_test_images()
    metric = BinaryBoundaryIoU()
    metric.update(preds, target)
    print(metric.compute())


if __name__ == '__main__':
    test_boundary_iou()
