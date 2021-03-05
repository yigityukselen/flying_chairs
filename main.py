import torch
from FlyingChairs import FlyingChairs,  Args
from pwcnet import PWCDCNet
import wandb
from itertools import chain
import flow_vis
import cv2
import time

wandb.init(project="flying_chairs", entity="yigityukselen")
def L2Loss(predicted_flows, target_flow, upsamle_size):
    total_loss = 0.0
    upsample = torch.nn.Upsample(size=upsamle_size, mode='bilinear', align_corners=False)
    for idx in range(len(predicted_flows)):
        upsampled_prediction = upsample(predicted_flows[idx])
        upsampled_prediction = upsampled_prediction * (upsamle_size[0] // predicted_flows[idx].shape[2])
        total_loss += torch.norm(upsampled_prediction - target_flow, p=2, dim=1).mean()
    return total_loss


def main():
    args = Args(crop_size=[384, 448], inference_size=[-1, -1])
    train_loader = torch.utils.data.DataLoader(FlyingChairs(args, is_cropped=False, root='./FlyingChairs_release/training/'), batch_size=8, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(FlyingChairs(args, is_cropped=False, root='./FlyingChairs_release/validation/'), batch_size=8, shuffle=False)
    model = PWCDCNet()
    device = torch.device("cuda")
    model.to(device)
    optimizer = None
    for epoch in range(1, 200):
        epoch_start_time = time.time()
        if epoch < 120:     
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        elif epoch > 120 and epoch < 160:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.000025)
            
        epoch_loss = 0.0
        hundred_iter_loss = 0.0
        for idx, data in enumerate(train_loader, 1):
            model.train()
            optimizer.zero_grad()
            left_images, right_images, flow = data[0].to(device), data[1].to(device), data[2].to(device)
            flow2, flow3, flow4, flow5, flow6 = model(torch.cat((left_images, right_images), 1))
            loss = L2Loss([flow2, flow3, flow4, flow5, flow6], flow, (384, 512))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            hundred_iter_loss += loss.item()
            if idx % 100 == 0:
                print("[Epoch {} - Iteration {}] Loss: {:.5f}".format(epoch, idx,  hundred_iter_loss / 100))
                print("--- %s seconds for one hundred iterations" % (time.time() - epoch_start_time))
                hundred_iter_loss = 0.0
        print("[Epoch {}] Loss: {:.5f}".format(epoch, epoch_loss / idx))
        print("Number of iteration is {}".format(idx))
        print("--- %s seconds for one epoch training" % (time.time() - epoch_start_time))
        wandb.log({"Epoch Number": epoch, "Mean Epoch Loss": epoch_loss / idx})
        with torch.no_grad():
            model.eval()
            validation_loss = 0.0
            rgb_images = []
            predicted_flow_images = []
            target_flow_images = []
            for idx, data in enumerate(validation_loader, 1):
                left_images, right_images, flow = data[0].to(device), data[1].to(device), data[2].to(device)
                flow2 = model(torch.cat((left_images, right_images), 1))
                validation_loss += L2Loss([flow2], flow, (384, 512))
                upsample = torch.nn.Upsample(size=(384, 512), mode='bilinear', align_corners=False)
                upsampled_flow2 = upsample(flow2)
                upsampled_flow2 = upsampled_flow2 * (384 // flow2.shape[2])
                if idx == 1:
                    for img_idx in range(8):
                        img_show = left_images[img_idx].permute(1, 2, 0).cpu().numpy()
                        img_show = img_show.clip(0, 1)
                        rgb_images.append(wandb.Image(img_show, grouping=3))
                        flowcolored_pred = flow_vis.flow_to_color(upsampled_flow2[img_idx, :2, :, :].permute(1, 2, 0).cpu().numpy())
                        predicted_flow_images.append(wandb.Image(flowcolored_pred))

                        flowcolored_gt = flow_vis.flow_to_color((flow[img_idx].permute(1, 2, 0)[:, :, :2]).cpu().numpy())
                        target_flow_images.append(wandb.Image(flowcolored_gt))
                    images_for_wandb = list(chain.from_iterable(zip(rgb_images, predicted_flow_images, target_flow_images)))
                    wandb.log({'examples':images_for_wandb}, commit=False)
            print("Validation loss: {:.5f}".format(validation_loss / idx))
            wandb.log({"Validation Loss":validation_loss/idx})
        model.train()
    torch.save(model.state_dict(), "./models/flying_chairs.pth")
if __name__ == "__main__":
    main()
