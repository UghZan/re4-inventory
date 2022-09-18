using UnityEngine;

public enum ItemSlot
{
    ALL,
    COMMON,
    ARMOR,
    MASK,
    WEAPON,
    ACCESSORY
}
[CreateAssetMenu(menuName = "Items/New Item")]
public class ItemInfo : ScriptableObject
{
    [Header("Visual Parameters")]
    public string itemDisplayName;
    public string itemDescription;
    public Sprite itemIcon;
    public Vector2Int itemSize;

    [Header("Inventory Parameters")]
    public float itemWeight;
    public ItemSlot itemSlot;
    public int itemStackSize = 99;
    public bool essential = false;
}

