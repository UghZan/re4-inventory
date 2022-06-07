using UnityEngine;

[System.Serializable]
public class ItemStack
{
    public ItemInfo item;
    public int amount;
    public Vector2Int invPos;
    public int rotated;

    public ItemStack()
    {
        item = null;
        amount = 0;
    }

    public void Rotate()
    {
        rotated++;
        if (rotated > 3) rotated = 0;
    }

    public void SetRotation(int rotation)
    {
        rotated = rotation;
    }

    public Vector2Int GetRotatedSize()
    {
        if (rotated == 0 || rotated == 2) return item.itemSize;
        else return new Vector2Int(item.itemSize.y, item.itemSize.x);
    }

    public ItemStack(ItemInfo _item, int _amount = 1)
    {
        item = _item;
        amount = _amount;
    }

    //update stack by a certain number (delta)
    //forcedSubstract will take the most out of the stack and will return amount left in delta
    //returns true if update was successful, false if stack isn't big enough/we got over max stack size
    public bool UpdateStack(ref int delta, bool forced = false)
    {
        if (delta > 0)
        {
            int diff = item.itemStackSize - amount;
            if (diff >= delta) //we don't go overboard
            {
                amount += delta;
                delta = 0;
                return true;
            }
            else //addition is too much
            {
                amount += diff;
                delta -= diff;
                return false;
            }
        }
        else
        {
            int diff = amount + delta;
            if (diff >= 0)
            {
                amount += delta;
                delta = 0;
                return true;
            }
            else
            {
                if (forced)
                    amount = 0;

                delta = -diff;
                return false;
            }
        }
    }

    public void SetAmount(int newAmount) { amount = newAmount; }

    public bool IsEmpty() => amount <= 0;

    public bool IsFull() => amount == item.itemStackSize;
}
