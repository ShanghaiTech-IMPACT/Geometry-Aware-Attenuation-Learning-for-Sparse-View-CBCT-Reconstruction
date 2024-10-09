
def gradient1_loss(volume_gt, volume_predict, loss_func):
    gdx_real = volume_gt[1:, :, :] - volume_gt[:-1, :, :]
    gdy_real = volume_gt[:, 1:, :] - volume_gt[:, :-1, :]
    gdz_real = volume_gt[:, :, 1:] - volume_gt[:, :, :-1]
    gdx_fake = volume_predict[1:, :, :] - volume_predict[:-1, :, :]
    gdy_fake = volume_predict[:, 1:, :] - volume_predict[:, :-1, :]
    gdz_fake = volume_predict[:, :, 1:] - volume_predict[:, :, :-1]
    gd_loss = loss_func(gdx_real, gdx_fake) + loss_func(gdy_real, gdy_fake) + loss_func(gdz_real, gdz_fake)
    return gd_loss
